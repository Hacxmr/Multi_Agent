import re

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample

from inspect_ai.scorer import (
    scorer,
    Score,
    accuracy,
)

from src.solvers.single_agent_solver import (
    single_agent_solver,
)

from src.solvers.debate_solver import (
    debate_solver,
)


def gsm8k_record_to_sample(record):

    return Sample(
        input=record["question"],
        target=record["answer"],
    )


def extract_final_answer(text):

    match = re.search(
        r"FINAL_ANSWER:\s*(-?\d+\.?\d*)",
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


@scorer(metrics=[accuracy()])
def gsm8k_scorer():

    async def score(state, target):

        prediction = extract_final_answer(
            state.output.completion
        )

        gold_numbers = re.findall(
            r"-?\d+\.?\d*",
            target.text,
        )

        gold = (
            gold_numbers[-1]
            if gold_numbers
            else None
        )

        correct = prediction == gold

        return Score(
            value=correct,
            answer=prediction,
            explanation=f"gold={gold}",
        )

    return score


@task
def gsm8k_single():

    dataset = hf_dataset(
        path="openai/gsm8k",
        name="main",
        split="test",
        sample_fields=gsm8k_record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=single_agent_solver(),
        scorer=gsm8k_scorer(),
    )


@task
def gsm8k_debate():

    dataset = hf_dataset(
        path="openai/gsm8k",
        name="main",
        split="test",
        sample_fields=gsm8k_record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=debate_solver(
            rounds=3,
            agents=3,
        ),
        scorer=gsm8k_scorer(),
    )