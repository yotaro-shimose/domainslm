from dataclasses import dataclass
from typing import Iterable, Literal

from agents import (
    MessageOutputItem,
    ReasoningItem,
    RunResult,
)
from loguru import logger
from openai.types.responses.response_output_message import Content as MessageContent
from openai.types.responses.response_reasoning_item import Content as ReasoningContent
from pydantic import BaseModel

type AgentItems = MessageOutputItem | ReasoningItem
type Role = Literal["user", "assistant", "system"]


class SimpleReasoningItem(BaseModel):
    role: Role
    content: str


class SimpleMessageItem(BaseModel):
    role: Role
    content: str


type SimpleItem = SimpleReasoningItem | SimpleMessageItem


def contents2text(contents: Iterable[ReasoningContent | MessageContent]) -> str:
    return "¥n".join([val.text for val in contents])  # type: ignore


@dataclass
class OutputWithItems[TOutput]:
    final_output: TOutput
    items: list[AgentItems]

    def simplified(self) -> list[SimpleReasoningItem | SimpleMessageItem]:
        simplified_items: list[SimpleReasoningItem | SimpleMessageItem] = []
        for item in self.items:
            if isinstance(item, MessageOutputItem):
                simplified_items.append(
                    SimpleMessageItem(
                        role=item.raw_item.role,
                        content=contents2text(item.raw_item.content),
                    )
                )
            elif isinstance(item, ReasoningItem):
                simplified_items.append(
                    SimpleReasoningItem(
                        role="assistant",
                        content="\n".join(
                            [summary.text for summary in item.raw_item.summary]
                        ),
                    )
                )
        return simplified_items


@dataclass
class RunResultWrapper[TOutput]:
    result: RunResult

    def output_with_reasoning(
        self,
    ) -> OutputWithItems[TOutput]:
        new_items = [
            item
            for item in self.result.new_items
            if isinstance(
                item,
                (
                    MessageOutputItem,
                    ReasoningItem,
                ),
            )
        ]
        if len(new_items) != len(self.result.new_items):
            logger.warning(
                "Warning: Some items were filtered out in output_with_reasoning."
            )
        return OutputWithItems[TOutput](
            final_output=self.final_output(),
            items=new_items,
        )

    def final_output(self) -> TOutput:
        return self.result.final_output
