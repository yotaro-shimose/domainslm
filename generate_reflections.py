from pathlib import Path
from typing import Awaitable

import polars as pl
from loguru import logger
from pydantic import BaseModel

from domainslm.async_util import gather_with_semaphore
from domainslm.openai_util.agent import AgentRunFailure, AgentsSDKModel
from domainslm.openai_util.runresult import SimpleReasoningItem
from domainslm.openai_util.tracing import setup_openai_tracing
from domainslm.spider.env import (
    EnvironmentError,
    SpiderEnvironment,
)
from domainslm.spider.reflection import (
    ReflectionInput,
    generate_reflected_response,
)
from domainslm.spider.sft_sample import SFTDataset, SFTSample
from domainslm.vllm import VLLMSetup
from sql_agent_rewrite import SQLAgent
from train_sql_agent import RL_TRAINING_CONFIG


async def process_one_sample(
    db_id: str,
    question: str,
    ground_truth: str,
    model: AgentsSDKModel,
    truncation_limit: int,
) -> tuple[bool, SFTSample] | None:
    with SpiderEnvironment.from_db_id(db_id, split="train") as env:
        try:
            db_schema = env.get_table_info()
            dialect = env.dialect()
            if len(db_schema) > truncation_limit:
                logger.warning(f"Schema length is too long: {len(db_schema)}")
                return None
        except EnvironmentError as e:
            logger.warning(f"Schema acquisition failed: {e}")
            return None
        agent = SQLAgent(
            model=model,
            db_schema=db_schema,
            dialect=dialect,
        )
        try:
            result = await agent.run_agent(question)
        except AgentRunFailure as e:
            logger.warning(f"Agent execution failed: {e.cause} - {str(e)}")
            return None

        try:
            env_ret = await env.evaluate(
                gold_query=ground_truth,
                predicted_query=result.final_output.query,
            )
        except EnvironmentError as e:
            logger.warning(f"Evaluation failed: {e}")
            return None

        if env_ret.match:
            simplified_items = result.simplified()
            if len(simplified_items) and isinstance(
                simplified_items[0], SimpleReasoningItem
            ):
                reasoning = simplified_items[0].content
                return bool(env_ret.match), SFTSample(
                    db_id=db_id,
                    question=question,
                    reasoning=reasoning,
                    gt_query=result.final_output.query,
                )

        try:
            reflection = await generate_reflected_response(
                model=model,
                reflection_input=ReflectionInput(
                    db_id=db_id,
                    question=question,
                    db_schema=db_schema,
                    behavior=result.simplified(),
                    gt=ground_truth,
                    exec_result=env_ret.execution_result,
                    gt_exec_result=env_ret.gold_execution_result,
                ),
            )
        except AgentRunFailure as e:
            logger.warning(f"Reflection generation failed: {e.cause} - {str(e)}")
            return None

    return bool(env_ret.match), SFTSample(
        db_id=db_id,
        question=question,
        reasoning=reflection.chain_of_thought,
        gt_query=reflection.revised_query,
    )


class GenerateReflectionParams(BaseModel):
    truncation_limit: int
    num_concurrency: int


async def main():
    param = GenerateReflectionParams(
        truncation_limit=20000,
        num_concurrency=80,
    )
    setup_openai_tracing()
    vllm_setup = VLLMSetup.qwen3()
    await vllm_setup.ensure_vllm_running()
    train_data = (
        pl.read_parquet(RL_TRAINING_CONFIG["data"]["train_files"])
        # .sample(100)
        .iter_rows(named=True)
    )

    tasks: list[Awaitable[tuple[bool, SFTSample] | None]] = []
    for row in train_data:
        db_id: str = row["db_id"]
        question: str = row["question"]
        ground_truth: str = row["query"]
        tasks.append(
            process_one_sample(
                db_id=db_id,
                question=question,
                ground_truth=ground_truth,
                model=vllm_setup,
                truncation_limit=param.truncation_limit,
            )
        )

    reflection_results = await gather_with_semaphore(
        tasks, param.num_concurrency, progressbar=True
    )
    valid_results = [result[1] for result in reflection_results if result is not None]
    matches = [int(result[0]) for result in reflection_results if result is not None]

    # Split into 10 train and rest (approx 10) eval
    train_results = valid_results[:-10]
    eval_results = valid_results[-10:]

    print(f"Total accuracy: {sum(matches) / len(matches)}")
    print(f"Total valid results: {len(valid_results)}")
    print(f"Train size: {len(train_results)}")
    print(f"Eval size: {len(eval_results)}")

    train_collection = SFTDataset(samples=train_results)
    eval_collection = SFTDataset(samples=eval_results)

    Path("data/reflection_train.json").write_text(
        train_collection.model_dump_json(indent=2)
    )
    Path("data/reflection_eval.json").write_text(
        eval_collection.model_dump_json(indent=2)
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
