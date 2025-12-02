#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Evaluate a trained SQL agent on the Spider dataset.

This script loads a trained SQL agent model and evaluates it on the Spider test
dataset, computing accuracy metrics and per-database breakdowns.

Usage:
    python evaluate_sql_agent.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct
    python evaluate_sql_agent.py --model ./checkpoints/my_model --num-samples 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Awaitable, Callable, Optional

import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import BaseModel

from domainslm.async_util import gather_with_semaphore
from domainslm.vllm import VLLMSetup
from sql_agent_rewrite import DBTask, SQLAgent, evaluate_query

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationExample(BaseModel):
    """Single evaluation result."""

    question: str
    generated_query: str
    ground_truth_query: str
    score: float
    db_id: str
    error: Optional[str] = None
    time_taken: float = 0.0


class SQLEvaluationResult(BaseModel):
    """Complete evaluation results."""

    examples: list[EvaluationExample]
    total_time: float
    accuracy: float
    correct: int
    total: int


async def setup_vllm_server(model_path: str, port: int = 5222) -> VLLMSetup:
    """Start vLLM server with trained model checkpoint.

    Args:
        model_path: Path to trained checkpoint or HuggingFace model name
        port: Port to run vLLM server on

    Returns:
        VLLMSetup instance with running server
    """
    logger.info(f"Setting up vLLM server with model: {model_path}")

    data_parallel_size = 1
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            data_parallel_size = min(2, device_count)
            logger.info(f"Using {data_parallel_size} GPUs for data parallelism")

    vllm_setup = VLLMSetup(
        model=model_path,
        port=port,
        reasoning_parser=None,  # SQL doesn't need reasoning parser
        data_parallel_size=data_parallel_size,
    )

    await vllm_setup.ensure_vllm_running()
    logger.info(f"vLLM server running at http://localhost:{port}")
    return vllm_setup


async def evaluate_single_task(
    task: DBTask,
    endpoint: str,
    model: str,
    spider_dir: str,
    use_test_db: bool = True,
) -> EvaluationExample:
    """Evaluate a single SQL generation task.

    Args:
        task: Database task containing question and ground truth
        endpoint: vLLM server endpoint URL
        model: Model name for API calls
        spider_dir: Path to Spider dataset directory
        use_test_db: Whether to use test_database (True) or database (False)

    Returns:
        EvaluationExample with results
    """
    question = task["question"]
    ground_truth = task["query"]
    db_id = task["db_id"]

    start_time = time.time()

    # Determine database path
    db_folder = "test_database" if use_test_db else "database"
    original_db_path = os.path.join(spider_dir, db_folder, db_id, f"{db_id}.sqlite")

    if not os.path.exists(original_db_path):
        error_msg = f"Database not found: {original_db_path}"
        logger.error(error_msg)
        return EvaluationExample(
            question=question,
            generated_query="",
            ground_truth_query=ground_truth,
            score=0.0,
            db_id=db_id,
            error=error_msg,
            time_taken=time.time() - start_time,
        )

    # Load schema if available
    schema_path = os.path.join(os.path.dirname(original_db_path), "schema.sql")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = f.read()
    else:
        schema = None

    # Create temporary database copy for this evaluation
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
        shutil.copyfile(original_db_path, db_path)

        try:
            # Create SQL agent instance
            agent = SQLAgent(
                db=f"sqlite:///{db_path}",
                endpoint=endpoint,
                model=model,
                api_key="dummy",
                db_schema=schema,
                table_info_truncate=2048,
            )

            # Run agent to generate query
            generated_query = await agent.run_agent(question)

        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            logger.exception(error_msg)
            return EvaluationExample(
                question=question,
                generated_query="",
                ground_truth_query=ground_truth,
                score=0.0,
                db_id=db_id,
                error=error_msg,
                time_taken=time.time() - start_time,
            )

    # Evaluate query on a fresh database copy
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
        shutil.copyfile(original_db_path, db_path)

        score = evaluate_query(
            query=generated_query,
            ground_truth=ground_truth,
            database=db_path,
            raise_on_error=False,
        )

    time_taken = time.time() - start_time

    return EvaluationExample(
        question=question,
        generated_query=generated_query,
        ground_truth_query=ground_truth,
        score=score,
        db_id=db_id,
        time_taken=time_taken,
    )


async def evaluate_sql_agent(
    model_path: str,
    test_data: list[DBTask],
    spider_dir: str,
    max_concurrent: int = 10,
    use_test_db: bool = True,
    vllm_port: int = 5222,
) -> SQLEvaluationResult:
    """Evaluate SQL agent on test dataset.

    Args:
        model_path: Path to trained model or HF model name
        test_data: List of test tasks
        spider_dir: Path to Spider dataset directory
        max_concurrent: Maximum concurrent evaluations
        use_test_db: Whether to use test_database (True) or database (False)
        vllm_port: Port for vLLM server

    Returns:
        SQLEvaluationResult with complete evaluation metrics
    """
    # Start vLLM server
    vllm_setup = await setup_vllm_server(model_path, port=vllm_port)
    endpoint = f"http://localhost:{vllm_port}/v1"
    model_name = vllm_setup.model

    logger.info(
        f"Evaluating {len(test_data)} examples with max_concurrent={max_concurrent}"
    )

    # Create evaluation tasks
    eval_start_time = time.time()
    tasks = [
        evaluate_single_task(
            task=task,
            endpoint=endpoint,
            model=model_name,
            spider_dir=spider_dir,
            use_test_db=use_test_db,
        )
        for task in test_data
    ]

    # Run evaluations concurrently with semaphore
    examples = await gather_with_semaphore(tasks, max_concurrent=max_concurrent)

    total_time = time.time() - eval_start_time

    # Compute overall metrics
    total = len(examples)
    correct = sum(1 for ex in examples if ex.score == 1.0)
    accuracy = correct / total if total > 0 else 0.0

    return SQLEvaluationResult(
        examples=examples,
        total_time=total_time,
        accuracy=accuracy,
        correct=correct,
        total=total,
    )


def compute_per_db_metrics(result: SQLEvaluationResult) -> dict[str, dict]:
    """Compute per-database accuracy metrics.

    Args:
        result: Evaluation results

    Returns:
        Dictionary mapping db_id to metrics (correct, total, accuracy)
    """
    per_db: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for example in result.examples:
        db_id = example.db_id
        per_db[db_id]["total"] += 1
        if example.score == 1.0:
            per_db[db_id]["correct"] += 1

    # Calculate accuracy for each database
    for db_id, metrics in per_db.items():
        metrics["accuracy"] = (
            metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
        )

    return dict(per_db)


def print_results(result: SQLEvaluationResult, per_db_metrics: dict[str, dict]) -> None:
    """Print evaluation results to console.

    Args:
        result: Evaluation results
        per_db_metrics: Per-database metrics
    """
    print("\n" + "=" * 80)
    print("SQL AGENT EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nOverall Accuracy: {result.accuracy:.2%}")
    print(f"Correct: {result.correct}/{result.total}")
    print(f"Total Time: {result.total_time:.2f}s")
    print(f"Avg Time per Query: {result.total_time / result.total:.2f}s")

    # Error analysis
    errors = [ex for ex in result.examples if ex.error is not None]
    if errors:
        print(f"\nErrors Encountered: {len(errors)}")
        error_types = defaultdict(int)
        for ex in errors:
            error_types[ex.error[:50]] += 1
        print("\nTop Error Types:")
        for error, count in sorted(
            error_types.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"  {error}... : {count}")

    # Per-database breakdown (top 10 and bottom 10)
    print("\n" + "-" * 80)
    print("PER-DATABASE ACCURACY")
    print("-" * 80)

    sorted_dbs = sorted(
        per_db_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    print("\nTop 10 Databases:")
    for db_id, metrics in sorted_dbs[:10]:
        print(
            f"  {db_id:20s}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})"
        )

    if len(sorted_dbs) > 10:
        print("\nBottom 10 Databases:")
        for db_id, metrics in sorted_dbs[-10:]:
            print(
                f"  {db_id:20s}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})"
            )

    print("=" * 80 + "\n")


def save_results(
    result: SQLEvaluationResult,
    per_db_metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Save detailed results to JSON file.

    Args:
        result: Evaluation results
        per_db_metrics: Per-database metrics
        output_path: Path to output JSON file
    """
    output_data = {
        "overall": {
            "accuracy": result.accuracy,
            "correct": result.correct,
            "total": result.total,
            "total_time": result.total_time,
            "avg_time_per_query": result.total_time / result.total,
        },
        "per_database": per_db_metrics,
        "examples": [ex.model_dump() for ex in result.examples],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Detailed results saved to {output_path}")


async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained SQL agent on Spider dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./actor/huggingface",
        help="Path to trained model checkpoint or HuggingFace model name",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dev.parquet",
        help="Path to test data parquet file (default: data/dev.parquet)",
    )
    parser.add_argument(
        "--spider-dir",
        type=str,
        default=None,
        help="Path to Spider dataset directory (default: from VERL_SPIDER_DATA_DIR env or 'data')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent evaluations (default: 10)",
    )
    parser.add_argument(
        "--use-train-db",
        action="store_true",
        help="Use training databases instead of test databases",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to save detailed results JSON (optional)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=5222,
        help="Port for vLLM server (default: 5222)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Determine Spider directory
    spider_dir = args.spider_dir or os.environ.get("VERL_SPIDER_DATA_DIR", "data")
    logger.info(f"Using Spider directory: {spider_dir}")

    # Load test data
    data_path = args.data
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Loading test data from {data_path}")
    df = pd.read_parquet(data_path)

    if args.num_samples is not None:
        df = df.head(args.num_samples)

    test_data: list[DBTask] = df.to_dict(orient="records")  # type: ignore
    logger.info(f"Loaded {len(test_data)} test examples")

    # Run evaluation
    result = await evaluate_sql_agent(
        model_path=args.model,
        test_data=test_data,
        spider_dir=spider_dir,
        max_concurrent=args.max_concurrent,
        use_test_db=not args.use_train_db,
        vllm_port=args.vllm_port,
    )

    # Compute per-database metrics
    per_db_metrics = compute_per_db_metrics(result)

    # Print results
    print_results(result, per_db_metrics)

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        save_results(result, per_db_metrics, output_path)


if __name__ == "__main__":
    asyncio.run(main())
