import os
import time
from pathlib import Path
from typing import Awaitable, Callable

import matplotlib.pyplot as plt
import seaborn as sns
from agents import Agent, Runner
from agents.tracing import set_tracing_export_api_key
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix

from benchmark.casehold import CaseHoldAnswer, CaseHoldQuestion, evaluate_casehold
from benchmark.unfair_tos import UnfairTosQuestion, UnfairTosResult, evaluate_unfair_tos
from domainslm.vllm import VLLMSetup


def visualize_confusion_matrix(result: UnfairTosResult, out_path: Path) -> None:
    # Confusion Matrix Visualization
    # For multi-label classification, we'll use the first ground truth label
    y_true = [g[0] if isinstance(g, list) and len(g) > 0 else g for g in result.gts]
    y_pred = result.answers

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualize with seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=range(cm.shape[1]),  # type: ignore
        yticklabels=range(cm.shape[0]),  # type: ignore
    )
    plt.title("Confusion Matrix - UnfairTOS Classification", fontsize=16, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")


def create_casehold_agent(
    model: str,
) -> tuple[Agent, Callable[[CaseHoldQuestion], Awaitable[int]]]:
    agent = Agent(
        name="casehold_agent",
        model=model,
        output_type=CaseHoldAnswer,
        instructions="""\
You are a legal expert. You will be given a legal text with a <HOLDING> placeholder and five candidate holding statements (holding_0 through holding_4).

Your task is to determine which holding statement best fills the <HOLDING> placeholder in the citing prompt.

Analyze the legal context carefully and select the holding that logically fits the citation.

Respond with the index (0-4) of the correct holding.
Response format: Output your answer as a JSON object with a single field "holding" containing the integer index (0-4) of the correct holding. Example: {"holding": 2}
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

Which holding (0-4) correctly fills the <HOLDING> placeholder?

"""

        result = await Runner.run(agent, input=prompt)
        answer = result.final_output_as(CaseHoldAnswer)
        return answer.holding

    return agent, run_agent


class UnfairToSAnswerResponse(BaseModel):
    answer: int


def create_unfair_tos_agent(
    model: str,
) -> tuple[Agent, Callable[[UnfairTosQuestion], Awaitable[int]]]:
    agent = Agent(
        name="unfair_tos_agent",
        model=model,
        output_type=UnfairToSAnswerResponse,
        instructions="""\
You are a legal expert analyzing Terms of Service clauses. You will be given a clause from a Terms of Service agreement and a list of categories.

Your task is to identify which categories describe the potentially unfair aspects of the clause. There may be multiple applicable categories.

Analyze the clause carefully and select all applicable category indices from the options provided.
Response format: Output your answer as a JSON object with a single field "answer" containing an integer index. Example: {"answer": 8}
""",
    )

    async def run_agent(question: UnfairTosQuestion) -> int:
        options_str = "\n".join(f"{i}: {opt}" for i, opt in enumerate(question.options))
        prompt = f"""Terms of Service clause:
{question.input}

Categories:
{options_str}

Which category indices describe the unfair aspects of this clause? Select all that apply."""

        result = await Runner.run(agent, input=prompt)
        answer = result.final_output_as(UnfairToSAnswerResponse).answer
        return answer

    return agent, run_agent


async def main():
    load_dotenv()
    set_tracing_export_api_key(os.environ["OPENAI_API_KEY"])
    vllm_setup = VLLMSetup.qwen3()
    if not await vllm_setup.is_vllm_running():
        raise ValueError("vLLM server is not running!")
    # model = vllm_setup.litellm_agentssdk_name()
    model = "gpt-5"

    # # CaseHOLD evaluation
    # _, casehold_run_agent = create_casehold_agent(model)

    # start_time = time.perf_counter()
    # result = await evaluate_casehold(casehold_run_agent, num_samples=100, max_concurrent=100)
    # elapsed_time = time.perf_counter() - start_time

    # correct = sum(1 for a, g in zip(result.answers, result.gts) if a == g)
    # total = len(result.answers)
    # accuracy = correct / total if total > 0 else 0.0

    # print("CaseHOLD Evaluation Results:")
    # print(f"  Correct: {correct}/{total}")
    # print(f"  Accuracy: {accuracy:.2%}")
    # print(f"  Runtime: {elapsed_time:.2f}s")
    # print(f"  Avg per sample: {elapsed_time / total:.2f}s")

    # UnfairTOS evaluation
    _, unfair_tos_run_agent = create_unfair_tos_agent(model)

    start_time = time.perf_counter()
    result = await evaluate_unfair_tos(
        unfair_tos_run_agent, num_samples=100, max_concurrent=100, none_frequency=10
    )
    elapsed_time = time.perf_counter() - start_time
    correct = sum(1 for a, g in zip(result.answers, result.gts) if a in g)

    total = len(result.answers)
    accuracy = correct / total if total > 0 else 0.0

    print("\nUnfairTOS Evaluation Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Runtime: {elapsed_time:.2f}s")
    print(f"  Avg per sample: {elapsed_time / total:.2f}s")

    out_path = Path("unfair_tos_confusion_matrix.png")
    visualize_confusion_matrix(result, out_path=out_path)
    print(f"\nConfusion matrix saved to: {out_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
