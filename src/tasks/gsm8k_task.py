from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.scorer import match

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
        metadata={
            "question": record["question"],
        },
    )


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
        scorer=match(),
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
        scorer=match(),
    )