from typing import Awaitable, Callable, Iterable, Self, TypedDict

from datasets import load_dataset
from pydantic import BaseModel

from domainslm.async_util import gather_with_semaphore


class UnfairTosResult(BaseModel):
    answers: list[int]
    gts: list[list[int]]


class UnfairTosRawExample(TypedDict):
    id: int
    input: str
    gold_index: list[int]
    options: list[str]


class UnfairTosQuestion(BaseModel):
    id: int
    input: str
    options: list[str]


class UnfairTosAnswer(BaseModel):
    gold_index: list[int]


class UnfairTosExample(BaseModel):
    id: int
    input: str
    gold_index: list[int]
    options: list[str]

    @classmethod
    def from_raw(cls, raw: UnfairTosRawExample) -> Self:
        return cls(
            id=raw["id"],
            input=raw["input"],
            gold_index=raw["gold_index"],
            options=[val for val in raw["options"]],
        )

    def to_qa(self) -> tuple[UnfairTosQuestion, UnfairTosAnswer]:
        return (
            UnfairTosQuestion(
                id=self.id,
                input=self.input,
                options=self.options,
            ),
            UnfairTosAnswer(gold_index=self.gold_index),
        )


def load_unfair_tos_dataset(none_frequency: int = 1) -> Iterable[UnfairTosExample]:
    """
    Load the UnfairTOS dataset.

    Args:
        none_frequency: Frequency to include samples with gold_index=[8] (not unfair).
                       For example, if none_frequency=10, only 1 out of 10 samples
                       with gold_index=[8] will be included (9 will be skipped).
                       Default is 1 (include all samples).
    """
    dataset: Iterable[UnfairTosRawExample] = load_dataset(
        "AdaptLLM/law-tasks",
        "UNFAIR_ToS",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )  # type: ignore

    none_counter = 0
    for raw in dataset:
        # Check if this is a "not unfair" sample (gold_index = [8])
        if raw["gold_index"] == [8]:
            none_counter += 1
            # Skip this sample if it's not the nth occurrence
            if none_counter % none_frequency != 0:
                continue

        yield UnfairTosExample.from_raw(raw)


async def evaluate_unfair_tos(
    agent: Callable[[UnfairTosQuestion], Awaitable[int]],
    num_samples: int,
    max_concurrent: int = 10,
    none_frequency: int = 1,
) -> UnfairTosResult:
    dataset = load_unfair_tos_dataset(none_frequency=none_frequency)
    tasks = []
    gts = []
    for i, example in enumerate(dataset):
        if i == num_samples:
            break
        question, answer = example.to_qa()
        tasks.append(agent(question))
        gts.append(answer.gold_index)

    answers = await gather_with_semaphore(tasks, max_concurrent=max_concurrent)

    return UnfairTosResult(answers=answers, gts=gts)
