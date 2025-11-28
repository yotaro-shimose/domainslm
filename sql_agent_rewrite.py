# Copyright (c) Microsoft. All rights reserved.

"""Sample code that demonstrates an SQL agent using LangGraph and LangChain,
trainable with Agent-lightning.

Adapted from https://python.langchain.com/docs/tutorials/sql_qa/
as well as https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""

from __future__ import annotations

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner as OpenAIRunner
from pydantic import BaseModel
import logging
import os
import shutil
import tempfile
import time
from typing import List, Optional, TypedDict, cast


import agentlightning as agl
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AnyMessage

from spider_eval.exec_eval import eval_exec_match

agl.logging.configure_logger()

logger = logging.getLogger(__name__)


class State(BaseModel):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[AnyMessage]


class SQLResponse(BaseModel):
    query: str


class SQLAgent:
    def __init__(
        self,
        db: str,
        endpoint: str,
        model: str,
        api_key: str | None,
        db_schema: str | None = None,
        table_info_truncate: int = 2048,
    ):
        self.db = SQLDatabase.from_uri(db)  # type: ignore
        self.db_schema = db_schema
        self.table_info_truncate = table_info_truncate
        self.agent = self.build_write_query_agent(
            endpoint=endpoint, model=model, api_key=api_key
        )

    def get_table_info(self) -> str:
        """Get the table information in a human-readable format."""
        try:
            table_info = self.db.get_table_info()
            if len(table_info) > self.table_info_truncate:
                table_info = (
                    table_info[: self.table_info_truncate] + "\n... (truncated)"
                )
            return table_info
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            if self.db_schema:
                if len(self.db_schema) > self.table_info_truncate:
                    return (
                        self.db_schema[: self.table_info_truncate] + "\n... (truncated)"
                    )
                return self.db_schema
            return "No schema available."

    def build_write_query_agent(
        self, endpoint: str, model: str, api_key: str | None
    ) -> Agent:
        agent = Agent(
            name="SQLWriteQueryAgent",
            model=OpenAIChatCompletionsModel(
                model=model,
                openai_client=AsyncOpenAI(
                    base_url=endpoint,
                    api_key=api_key,
                ),
            ),
            output_type=SQLResponse,
            instructions=f"""
You are an agent designed to interact with a SQL database.
     Given an input question, create a syntactically correct {self.db.dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{self.get_table_info()}

## Output Format ##

Respond in the following format:

```{self.db.dialect}
GENERATED QUERY
```
""".strip(),
        )
        return agent

    async def run_agent(self, question: str) -> str:
        result = await OpenAIRunner.run(
            self.agent,
            input=f"Question: {question}",
        )
        query = result.final_output_as(SQLResponse).query
        return query


class DBTask(TypedDict):
    question: str
    query: str
    db_id: str


class LitSQLAgent(agl.LitAgent[DBTask]):
    def __init__(
        self,
        trained_agents: Optional[str] = r"write",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.spider_dir = os.environ.get("VERL_SPIDER_DATA_DIR", "data")
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate

    async def rollout_async(
        self,
        task: DBTask,
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        if not isinstance(rollout, agl.AttemptedRollout):
            raise ValueError("Expected rollout to be of type AttemptedRollout")
        question = task["question"]
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        if rollout.mode == "train":
            original_db_path = os.path.join(
                self.spider_dir, "database", task["db_id"], task["db_id"] + ".sqlite"
            )
        else:
            original_db_path = os.path.join(
                self.spider_dir,
                "test_database",
                task["db_id"],
                task["db_id"] + ".sqlite",
            )
        ground_truth = task["query"]
        if not os.path.exists(original_db_path):
            logger.error(f"Database {original_db_path} does not exist. Skipping.")
            return None

        schema_path = os.path.join(os.path.dirname(original_db_path), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = f.read()
        else:
            logger.error("Schema file not found: %s", schema_path)
            schema = "No schema available."

        rollout_id = rollout.rollout_id

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)
            logger.info(f"[Rollout {rollout_id}] Question: {question}")
            logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

            # Run the agent
            agent = SQLAgent(
                "sqlite:///" + db_path,
                table_info_truncate=self.table_info_truncate,
                db_schema=schema,
                endpoint=llm.get_base_url(
                    rollout.rollout_id, rollout.attempt.attempt_id
                ),
                model=llm.model,
                api_key=llm.api_key,
            )
            try:
                result = await agent.run_agent(question)

            except Exception as e:
                logger.exception(
                    f"[Rollout {rollout_id}] Error during agent invocation: {e}"
                )
                return

            logger.info(f"[Rollout {rollout_id}] Generated Query: {result}")

        end_time_rollout = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)

            reward = evaluate_query(result, ground_truth, db_path, raise_on_error=False)
            logger.info("[Rollout %s] Reward: %s", rollout_id, reward)

        end_time_eval = time.time()

        logger.info(
            "[Rollout %s] Time taken for rollout: %.2f seconds",
            rollout_id,
            end_time_rollout - start_time,
        )
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds",
            rollout_id,
            end_time_eval - end_time_rollout,
        )

        return reward


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
    load_dotenv()
    debug_sql_agent()
