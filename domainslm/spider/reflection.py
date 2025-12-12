from pydantic import BaseModel

from domainslm.openai_util.agent import AgentsSDKModel, AgentWrapper
from domainslm.openai_util.runresult import SimpleItem
from domainslm.openai_util.types import ChatMLSample, ChatMLTextItem
from domainslm.spider.sql_agent import SQLResponse


class ReflectionInput(BaseModel):
    db_id: str
    question: str
    db_schema: str
    behavior: list[SimpleItem]
    gt: str
    exec_result: str
    gt_exec_result: str


class ReflectionOutput(BaseModel):
    analysis: str
    chain_of_thought: str
    revised_query: str


class ReflectionResult(BaseModel):
    input: ReflectionInput
    output: ReflectionOutput

    def as_md(self) -> str:
        behavior_str = "\n".join(
            [f"- ({item.role}) {item.content}" for item in self.input.behavior]
        )
        md = f"""
# Reflection Input
## Question
{self.input.question}
## Agent's Behavior
{behavior_str}
## Ground Truth Query
{self.input.gt}
## Execution Result of Generated Query
{self.input.exec_result}
## Execution Result of Ground Truth Query
{self.input.gt_exec_result}
## Reflection Output
## Analysis
{self.output.analysis}
## Refined Reasoning
{self.output.chain_of_thought}
## Revised Query
{self.output.revised_query}
""".strip()
        return md

    def as_chatml(self) -> ChatMLSample:
        messages = [
            ChatMLTextItem(role="user", content=f"Question:\n{self.input.question}"),
            ChatMLTextItem(
                role="assistant",
                content=f"<think>{self.output.chain_of_thought}</think>{SQLResponse(query=self.output.revised_query).model_dump_json()}",
            ),
        ]
        return ChatMLSample(messages=messages)


class ReflectionResults(BaseModel):
    results: list[ReflectionResult]


async def generate_reflected_response(
    model: AgentsSDKModel,
    reflection_input: ReflectionInput,
) -> ReflectionOutput:
    reflection_agent = AgentWrapper[ReflectionOutput].create(
        name="SQLReflectionAgent",
        instructions="""
You are an expert AI data annotator specializing in SQL generation.
Your task is to analyze a failed attempt and then generate a "Gold Standard" reasoning trace that leads to the correct answer.

You will be given:
1. The Question.
2. The Database Schema.
3. A failed Agent's Behavior (Reasoning + Query).
4. The Ground Truth Query (The correct answer).
5. Execution results.

### Objectives

1. **Analyze (Internal Critique):**
   Compare the Agent's Query with the Ground Truth. Identify strictly why the agent failed (e.g., wrong column, logic error, syntax error).

2. **Generate Refined Reasoning (The Core Task):**
   Construct a **brand new** Chain-of-Thought reasoning process that leads to the Ground Truth Query.
   
   **CRITICAL RULES for `refined_reasoning` field:**
   - **FIRST-PERSON SIMULATION:** Write as if you are the agent solving the problem for the **first time**.
   - **CLEAN SLATE:** Do NOT mention "the previous agent", "the mistake", "correction", "I should have", or "unlike the previous attempt".
   - **FORWARD LOOKING:** Start directly by analyzing the schema and the user question. Move step-by-step towards the solution.
   - The output must look exactly like a high-quality trace from a reasoning model (e.g., o1).

3. Based on the above, provide:
- An analysis about agent's reasoning process.
- The refined reasoning steps the agent should have taken. Note that this has to be self-contained so that this can directly be used as its supervised training data.
- A revised SQL query that better matches the ground truth (This should highly likely to be the same as the ground truth).
""",
        model=model,
        output_type=ReflectionOutput,
    )
    input_str = f"""
Question: 
{reflection_input.question}

Database Schema:
{reflection_input.db_schema}

Agent's Reasoning and Query:
{"\n".join([item.content for item in reflection_input.behavior])}

Ground Truth Query: {reflection_input.gt}

Execution Results of Generated Query: {reflection_input.exec_result}

Execution Results of Ground Truth Query: {reflection_input.gt_exec_result}
""".strip()
    ret = await reflection_agent.run(input=input_str)

    return ret.final_output()
