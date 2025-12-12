import asyncio
import os
from asyncio import timeout
from dataclasses import dataclass
from typing import Any, Literal, Self, Sequence

from agents import (
    Agent,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    Tool,
    TResponseInputItem,
    UserError,
)
from agents.run import DEFAULT_MAX_TURNS
from pydantic import BaseModel

from domainslm.openai_util.runresult import RunResultWrapper
from domainslm.vllm import VLLMSetup

from litellm import ContextWindowExceededError

type AgentsSDKModel = str | OpenAIChatCompletionsModel | VLLMSetup


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
        tools: Sequence[Tool] | None = None,
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
            tools=list(tools) if tools is not None else [],
            **kwargs,
        )
        return cls(agent=agent)

    async def run(
        self,
        input: str | list[TResponseInputItem],
        *,
        context: Any | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        time_out_seconds: float = 120.0,
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
        except ContextWindowExceededError as e:
            raise AgentRunFailure(
                str(e),
                cause="ContextWindowExceededError",
            )
        return RunResultWrapper[type(result.final_output)](result=result)
