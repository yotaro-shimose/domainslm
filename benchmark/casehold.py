from typing import Awaitable, Callable, Iterable, Self, TypedDict

from datasets import load_dataset
from pydantic import BaseModel

from domainslm.async_util import gather_with_semaphore


class CaseHoldResult(BaseModel):
    answers: list[int]
    gts: list[int]


class CaseHoldRawExample(TypedDict):
    example_id: int
    citing_prompt: str
    holding_0: str
    holding_1: str
    holding_2: str
    holding_3: str
    holding_4: str
    label: str


class CaseHoldQuestion(BaseModel):
    example_id: int
    citing_prompt: str
    holding_0: str
    holding_1: str
    holding_2: str
    holding_3: str
    holding_4: str


class CaseHoldAnswer(BaseModel):
    holding: int


class CaseHoldExample(BaseModel):
    example_id: int
    citing_prompt: str
    holding_0: str
    holding_1: str
    holding_2: str
    holding_3: str
    holding_4: str
    label: int

    @classmethod
    def from_raw(cls, raw: CaseHoldRawExample) -> Self:
        return cls(
            example_id=raw["example_id"],
            citing_prompt=raw["citing_prompt"],
            holding_0=raw["holding_0"],
            holding_1=raw["holding_1"],
            holding_2=raw["holding_2"],
            holding_3=raw["holding_3"],
            holding_4=raw["holding_4"],
            label=int(raw["label"]),
        )

    def to_qa(self) -> tuple[CaseHoldQuestion, CaseHoldAnswer]:
        return (
            CaseHoldQuestion(
                example_id=self.example_id,
                citing_prompt=self.citing_prompt,
                holding_0=self.holding_0,
                holding_1=self.holding_1,
                holding_2=self.holding_2,
                holding_3=self.holding_3,
                holding_4=self.holding_4,
            ),
            CaseHoldAnswer(holding=self.label),
        )


async def evaluate_casehold(
    agent: Callable[[CaseHoldQuestion], Awaitable[int]],
    num_samples: int,
    max_concurrent: int = 10,
) -> CaseHoldResult:
    dataset: Iterable[CaseHoldRawExample] = load_dataset(
        "casehold/casehold", split="train", streaming=True, trust_remote_code=True
    )  # type: ignore
    tasks = []
    gts = []
    for i, raw in enumerate(dataset):
        if i == num_samples:
            break
        example = CaseHoldExample.from_raw(raw)
        question, answer = example.to_qa()
        tasks.append(agent(question))
        gts.append(answer.holding)

    answers = await gather_with_semaphore(tasks, max_concurrent=max_concurrent)

    return CaseHoldResult(answers=answers, gts=gts)


async def main():
    import time

    from agents import Agent, Runner

    from domainslm.vllm import VLLMSetup

    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    vllm_setup = VLLMSetup(model=model_name)
    await vllm_setup.ensure_vllm_running()

    agent = Agent(
        name="casehold_agent",
        model=vllm_setup.litellm_agentssdk_name(),
        output_type=CaseHoldAnswer,
        instructions="""\
You are a legal expert. You will be given a legal text with a <HOLDING> placeholder and five candidate holding statements (holding_0 through holding_4).

Your task is to determine which holding statement best fills the <HOLDING> placeholder in the citing prompt.

Analyze the legal context carefully and select the holding that logically fits the citation.

Respond with the index (0-4) of the correct holding.
""",
    )

    async def run_agent(question: CaseHoldQuestion) -> int:
        prompt = f"""Citing prompt (with <HOLDING> placeholder):
{question.citing_prompt}

Candidate holdings:
0: {question.holding_0}
1: {question.holding_1}
2: {question.holding_2}
3: {question.holding_3}
4: {question.holding_4}

Which holding (0-4) correctly fills the <HOLDING> placeholder?"""

        result = await Runner.run(agent, input=prompt)
        answer = result.final_output_as(CaseHoldAnswer)
        return answer.holding

    start_time = time.perf_counter()
    result = await evaluate_casehold(run_agent, num_samples=100, max_concurrent=10)
    elapsed_time = time.perf_counter() - start_time

    correct = sum(1 for a, g in zip(result.answers, result.gts) if a == g)
    total = len(result.answers)
    accuracy = correct / total if total > 0 else 0.0

    print("CaseHOLD Evaluation Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Runtime: {elapsed_time:.2f}s")
    print(f"  Avg per sample: {elapsed_time / total:.2f}s")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
