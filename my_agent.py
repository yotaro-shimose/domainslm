import subprocess
import time
from typing import Self

import agentlightning as agl
import httpx
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, function_tool
from agents import Runner as OpenAIRunner
from openai import AsyncOpenAI
from pydantic import BaseModel, InstanceOf


class VLLMSetup(BaseModel):
    model: str
    port: int = 5222
    api_key: str = "your_api_key_here"
    vllm_process: InstanceOf[subprocess.Popen | None] = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    async def is_vllm_running(self) -> bool:
        url = f"{self.base_url}/health"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                return response.status_code == 200
            except httpx.ConnectError:
                return False

    def launch_vllm_server(self) -> subprocess.Popen:
        vllm_process = subprocess.Popen(
            [
                "vllm",
                "serve",
                self.model,
                "--port",
                str(self.port),
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
                "--reasoning-parser",
                "deepseek_r1",
                "--api-key",
                self.api_key,
            ]
        )
        self.vllm_process = vllm_process
        return vllm_process

    def wait_for_server(self, timeout: int = 180) -> None:
        url = f"{self.base_url[:-3]}/health"
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


@function_tool
def double_tool(x: int) -> int:
    print(f"Doubling {x}")
    return x * 2


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
                    "model": cls.litellm_model(vllm_setup.model),
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

    @classmethod
    def litellm_model(cls, model: str) -> str:
        return f"hosted_vllm/{model}"

    def get_openai_clent(self) -> AsyncOpenAI:
        llm = self.llm_proxy.as_resource()
        return AsyncOpenAI(
            base_url=llm.endpoint,
            api_key=self.vllm_setup.api_key,
        )

    async def ensure_vllm_running(self) -> None:
        await self.vllm_setup.ensure_vllm_running()

    def chat_completions_model(self) -> OpenAIChatCompletionsModel:
        return OpenAIChatCompletionsModel(
            model=self.litellm_model(self.vllm_setup.model),
            openai_client=self.get_openai_clent(),
        )


async def main() -> None:
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    vllm_setup = VLLMSetup(model=model_name)
    proxy = await ProxiedVLLMSetup.create(vllm_setup)
    await proxy.ensure_vllm_running()

    my_agent = Agent(
        name="MyVLLMAgent",
        model=proxy.chat_completions_model(),
        instructions="You are a helpful assistant that can double numbers using the provided tool. Then response with the acquired result to user.",
        # tools=[double_tool],
        # model_settings=ModelSettings(tool_choice="required"),
    )
    runner = OpenAIRunner()
    ret = await runner.run(
        my_agent,
        input="What is 15 doubled?",
    )
    print("Agent output:", ret.final_output_as(str))
    pass
    # TODO: try visual input


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
