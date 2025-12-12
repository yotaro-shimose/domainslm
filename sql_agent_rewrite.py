# Copyright (c) Microsoft. All rights reserved.

"""Sample code that demonstrates an SQL agent using LangGraph and LangChain,
trainable with Agent-lightning.

Adapted from https://python.langchain.com/docs/tutorials/sql_qa/
as well as https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, cast

import agentlightning as agl
import pandas as pd
from agents import OpenAIChatCompletionsModel
from dotenv import load_dotenv
from openai import AsyncOpenAI

from domainslm.openai_util.agent import AgentRunFailure
from domainslm.spider.env import SpiderEnvironment
from domainslm.spider.sft_sample import DBTask
from domainslm.spider.sql_agent import SQLAgent
from spider_eval.exec_eval import eval_exec_match

agl.logging.configure_logger()

logger = logging.getLogger(__name__)


class LitSQLAgent(agl.LitAgent[DBTask]):
    def __init__(
        self,
        trained_agents: Optional[str] = r"write",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        table_info_truncate: int = 20000,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.spider_dir = os.environ.get("VERL_SPIDER_DATA_DIR", "data")
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate

    async def rollout_async(
        self,
        task: DBTask,
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        if not isinstance(rollout, agl.AttemptedRollout):
            raise ValueError("Expected rollout to be of type AttemptedRollout")
        question = task["question"]
        db_id = task["db_id"]
        ground_truth = task["query"]
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        # Run the agent
        model = OpenAIChatCompletionsModel(
            model=llm.model,
            openai_client=AsyncOpenAI(
                base_url=llm.get_base_url(
                    rollout.rollout_id, rollout.attempt.attempt_id
                ),
                api_key=llm.api_key,
            ),
        )
        with SpiderEnvironment.from_db_id(
            db_id, split="train" if rollout.mode == "train" else "test"
        ) as env:
            try:
                db_schema = env.get_table_info()
                dialect = env.dialect()
                if len(db_schema) > self.table_info_truncate:
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
            return env_ret.match


def evaluate_query(
    query: str, ground_truth: str, database: str, raise_on_error: bool = True
) -> float:
    # TODO(yuge): Maybe we can evaluate intermediate queries and assign more precise rewards.

    # included in the original evaluation script
    # query = query.replace("value", "1")

    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")

        # Parameters following the default setting
        exec_score = eval_exec_match(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        if exec_score == 1:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.exception(f"Error evaluating query: {e}")
            return 0.0


def debug_sql_agent():
    spider_dev_data_path = os.path.join(
        os.environ.get("VERL_SPIDER_DATA_DIR", "data"), "dev.parquet"
    )
    if not os.path.exists(spider_dev_data_path):
        raise FileNotFoundError(
            f"Spider dev data file {spider_dev_data_path} does not exist."
        )
    df = pd.read_parquet(spider_dev_data_path).head(10)  # type: ignore
    lines = cast(List[DBTask], df.to_dict(orient="records"))

    print("Debug data:", lines)

    trainer = agl.Trainer(
        n_runners=4,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint="https://api.openai.com/v1",
                model="gpt-5-nano",
                sampling_parameters={"temperature": 0.7},
                api_key=os.environ["OPENAI_API_KEY"],
            )
        },
    )
    trainer.dev(LitSQLAgent(), lines)


if __name__ == "__main__":
    # Hello tracer!
    load_dotenv()
    debug_sql_agent()
