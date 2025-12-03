import os
import subprocess
import time
from typing import Self

import agentlightning as agl
import httpx
import torch
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from pydantic import BaseModel, InstanceOf


class LiteLLMModelName(BaseModel):
    model_name: str


class VLLMSetup(BaseModel):
    model: str
    port: int = 5222
    max_model_len: int = 32768
    api_key: str = "your_api_key_here"
    vllm_process: InstanceOf[subprocess.Popen | None] = None
    data_parallel_size: int | None = None
    reasoning_parser: str | None = None

    @classmethod
    def qwen3(cls, **kwargs) -> Self:
        return cls(
            model="Qwen/Qwen3-4B",
            reasoning_parser="deepseek_r1",
            **kwargs,
        )

    @classmethod
    def qwen3_reasoning(cls, **kwargs) -> Self:
        return cls(
            model="Qwen/Qwen3-4B-Thinking-2507",
            reasoning_parser="deepseek_r1",
            **kwargs,
        )

    @classmethod
    def phi4(cls, **kwargs) -> Self:
        return cls(model="microsoft/Phi-4-mini-instruct", **kwargs)

    @classmethod
    def phi4_reasoning(cls, **kwargs) -> Self:
        return cls(
            model="microsoft/Phi-4-mini-reasoning",
            # reasoning_parser="deepseek_r1",
            **kwargs,
        )

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    async def is_vllm_running(self) -> bool:
        url = f"{self.base_url}/health"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                return response.status_code == 200
            except httpx.ConnectError:
                return False

    def launch_vllm_server(self) -> subprocess.Popen:
        commands: list[str] = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--max-model-len",
            str(self.max_model_len),
        ]
        if self.reasoning_parser is not None:
            commands.extend(
                [
                    "--reasoning-parser",
                    self.reasoning_parser,
                ]
            )
        if self.data_parallel_size is None:
            device_count = torch.cuda.is_available()
        else:
            device_count = self.data_parallel_size
        if device_count > 1:
            commands.extend(
                [
                    "--data-parallel-size",
                    str(self.data_parallel_size),
                ]
            )
        vllm_process = subprocess.Popen(commands)
        self.vllm_process = vllm_process
        return vllm_process

    def wait_for_server(self, timeout: int = 180) -> None:
        url = f"{self.base_url}/health"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(url, headers=headers)
                if response.status_code == 200:
                    return
            except httpx.ConnectError:
                time.sleep(5)
        raise TimeoutError("VLLM server did not start within the given timeout.")

    async def ensure_vllm_running(self) -> None:
        # Setup vLLM server if not running
        if not await self.is_vllm_running():
            print("VLLM server not running. Launching...")
            process = self.launch_vllm_server()
            try:
                self.wait_for_server()
                print("VLLM server is up and running.")
            except TimeoutError as e:
                process.terminate()
                raise e
        else:
            print("VLLM server is already running.")

    def get_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
        )

    @classmethod
    def litellm_model(cls, model: str) -> str:
        return f"hosted_vllm/{model}"

    def chat_completions_model(self) -> OpenAIChatCompletionsModel:
        return OpenAIChatCompletionsModel(
            model=self.litellm_model(self.model),
            openai_client=self.get_openai_client(),
        )

    def litellm_agentssdk_name(self) -> LiteLLMModelName:
        os.environ["OPENAI_API_KEY"] = (
            "dummy"  # Still needed for the SDK initialization
        )
        os.environ["HOSTED_VLLM_API_BASE"] = "http://localhost:5222/v1"
        os.environ["HOSTED_VLLM_API_KEY"] = "dummy"
        return LiteLLMModelName(model_name=f"litellm/{self.litellm_model(self.model)}")


class ProxiedVLLMSetup(BaseModel):
    proxy_port: int
    vllm_setup: VLLMSetup
    llm_proxy: InstanceOf[agl.LLMProxy]

    @classmethod
    async def create(cls, vllm_setup: VLLMSetup, proxy_port: int = 5223) -> Self:
        # Try using the vLLM server
        store = agl.InMemoryLightningStore()
        llm_proxy = agl.LLMProxy(port=proxy_port, store=store)

        model_list = [
            {
                "model_name": vllm_setup.model,
                "litellm_params": {
                    "model": vllm_setup.litellm_model(vllm_setup.model),
                    "api_base": vllm_setup.base_url,
                    "api_key": vllm_setup.api_key,
                },
            }
        ]
        llm_proxy.update_model_list(model_list)  # type: ignore
        # If the proxy is not running, it will start automatically.
        llm_proxy.restart()
        # Add the proxy as a resource to the store so that the runners can access it via URL.
        _resource_update = await store.add_resources(
            {"main_llm": llm_proxy.as_resource()}
        )

        return cls(
            vllm_setup=vllm_setup,
            proxy_port=proxy_port,
            llm_proxy=llm_proxy,
        )

    def get_openai_client(self) -> AsyncOpenAI:
        llm = self.llm_proxy.as_resource()
        return AsyncOpenAI(
            base_url=llm.endpoint,
            api_key=self.vllm_setup.api_key,
        )

    async def ensure_vllm_running(self) -> None:
        await self.vllm_setup.ensure_vllm_running()

    def chat_completions_model(self) -> OpenAIChatCompletionsModel:
        return OpenAIChatCompletionsModel(
            model=self.vllm_setup.litellm_model(self.vllm_setup.model),
            openai_client=self.get_openai_client(),
        )
