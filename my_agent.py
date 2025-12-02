from agents import Agent, Runner as OpenAIRunner

from domainslm.vllm import ProxiedVLLMSetup, VLLMSetup


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
