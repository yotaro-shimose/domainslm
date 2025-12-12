from typing import TypedDict

import polars as pl
from datasets import Dataset
from pydantic import BaseModel

from domainslm.openai_util.types import ChatMLSample, ChatMLTextItem
from domainslm.spider.reflection import SQLResponse


class DBTask(TypedDict):
    question: str
    query: str
    db_id: str


class SFTSample(BaseModel):
    db_id: str
    question: str
    reasoning: str
    gt_query: str

    def as_chatml(self) -> ChatMLSample:
        messages = [
            ChatMLTextItem(role="user", content=f"Question:\n{self.question}"),
            ChatMLTextItem(
                role="assistant",
                content=f"<think>{self.reasoning}</think>{SQLResponse(query=self.gt_query).model_dump_json()}",
            ),
        ]
        return ChatMLSample(messages=messages)


class SFTDataset(BaseModel):
    samples: list[SFTSample]

    def as_prompt_completion(self) -> Dataset:
        prompt = []
        completion = []
        for sample in self.samples:
            messages = sample.as_chatml()["messages"]
            prompt.append([messages[0]])
            completion.append([messages[1]])
        dataframe = pl.DataFrame({"prompt": prompt, "completion": completion})
        return Dataset.from_polars(dataframe)

    def as_rl_dataset(self) -> list[DBTask]:
        tasks = []
        for sample in self.samples:
            tasks.append(
                {
                    "db_id": sample.db_id,
                    "question": sample.question,
                    "query": sample.gt_query,
                }
            )
        return tasks
