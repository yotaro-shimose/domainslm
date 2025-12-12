from pydantic import BaseModel

from domainslm.openai_util.agent import AgentsSDKModel, AgentWrapper
from domainslm.openai_util.runresult import OutputWithItems


class SQLResponse(BaseModel):
    query: str


class SQLAgent:
    def __init__(
        self,
        model: AgentsSDKModel,
        db_schema: str,
        dialect: str,
        table_info_truncate: int = 2048,
    ):
        self.db_schema = db_schema
        self.table_info_truncate = table_info_truncate
        self.agent = self.build_write_query_agent(
            model=model, table_info=db_schema, dialect=dialect
        )

    def build_write_query_agent(
        self, model: AgentsSDKModel, table_info: str, dialect: str
    ) -> AgentWrapper[SQLResponse]:
        wrapper = AgentWrapper.create(
            name="SQLWriteQueryAgent",
            instructions=f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info[: self.table_info_truncate]}

## Output Format ##
Your output should be structured as a following JSON format:
{{
    "query": "GENERATED QUERY"
}}
""".strip(),
            model=model,
            output_type=SQLResponse,
        )

        return wrapper

    async def run_agent(self, question: str) -> OutputWithItems[SQLResponse]:
        result = await self.agent.run(
            input=question,
        )

        return result.output_with_reasoning()
