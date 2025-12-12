import polars as pl
from loguru import logger

from domainslm.async_util import gather_with_semaphore
from domainslm.openai_util.agent import AgentRunFailure, AgentsSDKModel
from domainslm.openai_util.tracing import setup_openai_tracing
from domainslm.spider.env import ExecutionEvalResult, SpiderEnvironment, SpiderSplit
from domainslm.spider.sql_agent import SQLAgent
from domainslm.vllm import VLLMSetup


async def evaluate(
    question: str,
    ground_truth: str,
    model: AgentsSDKModel,
    split: SpiderSplit,
    db_id: str,
    truncation_limit: int,
) -> ExecutionEvalResult | None:
    with SpiderEnvironment.from_db_id(
        db_id, split=split
    ) as env:
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
        return env_ret


async def main() -> None:
    # Example usage
    setup_openai_tracing()
    model = VLLMSetup(
        model="qwen3-4b-sql-merged",
        reasoning_parser="deepseek_r1",
        data_parallel_size=2,
    )
    await model.ensure_vllm_running()
    test_df = pl.read_parquet("data/test_dev_500.parquet")
    results: list[ExecutionEvalResult | None] = []
    truncation_limit = 20000

    results = await gather_with_semaphore(
        [
            evaluate(
                question=row["question"],
                ground_truth=row["query"],
                model=model,
                split="test",
                db_id=row["db_id"],
                truncation_limit=truncation_limit,
            )
            for row in test_df.iter_rows(named=True)
        ],
        max_concurrent=100,
    )
    valid_results = [r for r in results if r is not None]
    logger.info(f"Total evaluated samples: {len(valid_results)}")
    logger.info(
        f"Overall Match Rate: {sum(r.match for r in valid_results) / len(valid_results):.4f}"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
