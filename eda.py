import os

from agents import Agent, Runner, set_tracing_export_api_key
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel

from domainslm.async_util import gather_with_semaphore
from domainslm.vllm import VLLMSetup

# Full FreeLaw from original Pile (streaming recommended due to size)
dataset = load_dataset(
    "timaeus/pile-freelaw", split="train", streaming=True, trust_remote_code=True
)

# Filter for FreeLaw only
freelaw = dataset.filter(lambda x: x["meta"]["pile_set_name"] == "FreeLaw")


class QAExample(BaseModel):
    question: str
    answer: str


class RQAExample(BaseModel):
    question: str
    answer: str
    reference: str


async def generate_qa(model: str, corpora: str) -> RQAExample:
    agent = Agent(
        name="LegalQAAgent",
        model=model,
        instructions="""
You are a legal school teacher. Given the legal text, generate a question and answer pair that tests comprehension of the material.
Please make sure:
- Question do not refer to the provided legal text.
- Answers can be directly inferred from the legal text.
""",
        output_type=QAExample,
    )
    ret = await Runner.run(agent, input=corpora)
    qa = ret.final_output_as(QAExample)
    return RQAExample(
        question=qa.question,
        answer=qa.answer,
        reference=corpora,
    )


async def main():
    load_dotenv()
    set_tracing_export_api_key(os.environ["OPENAI_API_KEY"])
    vllm_setup = VLLMSetup.qwen3()
    if not vllm_setup.is_vllm_running():
        raise ValueError("Server is not running")

    model = vllm_setup.litellm_agentssdk_name()
    # model = "gpt-5-mini"

    # Collect examples to process
    examples = []
    for idx, example in enumerate(freelaw):
        examples.append(example["text"])
        if idx >= 5:  # Limit to first 6 examples for brevity
            break

    # Process in parallel with concurrency limit
    qa_results = await gather_with_semaphore(
        (generate_qa(model, text) for text in examples), max_concurrent=3
    )

    for qa in qa_results:
        print("Q:", qa.question)
        print("A:", qa.answer)
        print("-" * 40)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
