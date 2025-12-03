from asyncio import timeout
import asyncio
from dataclasses import dataclass
import os
from typing import Iterable, Literal, Self
from loguru import logger
from dotenv import load_dotenv
from agents import (
    Agent,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunResult,
    TResponseInputItem,
    UserError,
    set_tracing_export_api_key,
    Runner,
    MessageOutputItem,
    ReasoningItem,
)
from agents.run import DEFAULT_MAX_TURNS
from pydantic import BaseModel
from domainslm.vllm import VLLMSetup
from openai.types.responses.response_reasoning_item import Content as ReasoningContent
from openai.types.responses.response_output_message import Content as MessageContent

type AgentsSDKModel = str | OpenAIChatCompletionsModel | VLLMSetup
type AgentItems = MessageOutputItem | ReasoningItem
type Role = Literal["user", "assistant", "system"]


class SimpleReasoningItem(BaseModel):
    role: Role
    content: str


class SimpleMessageItem(BaseModel):
    role: Role
    content: str


def contents2text(contents: Iterable[ReasoningContent | MessageContent]) -> str:
    return "¥n".join([val.text for val in contents])  # type: ignore


@dataclass
class OutputWithItems[TOutput]:
    final_output: TOutput
    items: list[AgentItems]

    def simplified(self) -> list[SimpleReasoningItem | SimpleMessageItem]:
        simplified_items: list[SimpleReasoningItem | SimpleMessageItem] = []
        for item in self.items:
            if item.raw_item.content is None:
                continue
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
                        role="system", content=contents2text(item.raw_item.content)
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


class AgentRunFailure(BaseException):
    def __init__(
        self,
        message: str,
        cause: Literal[
            "ModelBehaviourError", "Timeout", "MaxTurnsExceeded", "UserError"
        ],
    ):
        super().__init__(message)
        self.cause = cause


@dataclass
class AgentWrapper[TOutput: BaseModel | str]:
    agent: Agent

    @classmethod
    def create(
        cls,
        name: str,
        instructions: str,
        model: str | OpenAIChatCompletionsModel | VLLMSetup,
        model_settings: ModelSettings | None = None,
        output_type: type[TOutput] | None = None,
    ) -> Self:
        if isinstance(model, (str, OpenAIChatCompletionsModel)):
            agents_sdk_model = model
        elif isinstance(model, VLLMSetup):
            if os.getenv("HOSTED_VLLM_API_BASE") is None:
                os.environ["OPENAI_API_KEY"] = (
                    "dummy"  # Still needed for the SDK initialization
                )
                os.environ["HOSTED_VLLM_API_BASE"] = "http://localhost:5222/v1"
                os.environ["HOSTED_VLLM_API_KEY"] = "dummy"
            agents_sdk_model = f"litellm/{model.litellm_model(model.model)}"
        else:
            raise ValueError("Unsupported model type")
        kwargs = {}
        if model_settings is not None:
            kwargs["model_settings"] = model_settings
        agent = Agent(
            name=name,
            instructions=instructions,
            model=agents_sdk_model,
            output_type=output_type,
            **kwargs,
        )
        return cls(agent=agent)

    async def run(
        self,
        input: str | list[TResponseInputItem],
        *,
        context: None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        time_out_seconds: float = 60.0,
    ) -> RunResultWrapper[TOutput]:
        try:
            async with timeout(time_out_seconds):
                result = await Runner.run(
                    self.agent,
                    input=input,
                    context=context,
                    max_turns=max_turns,
                )
        except asyncio.TimeoutError as e:
            raise AgentRunFailure(
                str(e),
                cause="Timeout",
            )
        except ModelBehaviorError as e:
            raise AgentRunFailure(
                str(e),
                cause="ModelBehaviourError",
            )
        except MaxTurnsExceeded as e:
            raise AgentRunFailure(
                str(e),
                cause="MaxTurnsExceeded",
            )
        except UserError as e:
            raise AgentRunFailure(
                str(e),
                cause="UserError",
            )
        return RunResultWrapper[type(result.final_output)](result=result)


def setup_openai_tracing():
    load_dotenv()
    set_tracing_export_api_key(os.environ["OPENAI_API_KEY"])
