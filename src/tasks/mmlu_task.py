import re

from inspect_ai import Task, task

from inspect_ai.dataset import (
    hf_dataset,
    Sample,
)

from inspect_ai.scorer import (
    scorer,
    Score,
    accuracy,
)

from src.solvers.single_agent_solver import (
    single_agent_solver,
)

from src.solvers.majority_vote import (
    majority_vote_solver,
)


def extract_mcq_answer(text):

    if text is None:
        return None

    matches = re.findall(

        r"FINAL_ANSWER:\s*([A-D])",

        str(text),

        re.IGNORECASE,
    )

    if matches:

        return matches[-1].upper()

    return None


def format_question(record):

    choices = record["choices"]

    return f"""
Question:
{record["question"]}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Respond ONLY with:

FINAL_ANSWER: A

(or B/C/D)
""".strip()


def mmlu_record_to_sample(record):

    answer_idx = int(
        record["answer"]
    )

    answer_letter = [

        "A",
        "B",
        "C",
        "D",
    ][answer_idx]

    return Sample(

        input=format_question(
            record
        ),

        target=answer_letter,
    )


@scorer(metrics=[accuracy()])
def mmlu_scorer():

    async def score(state, target):

        raw_output = (
            state.output.completion
        )

        prediction = extract_mcq_answer(
            raw_output
        )

        gold = target.text

        correct = (
            prediction == gold
        )

        print("\n===================")

        print(
            "PRED:",
            prediction
        )

        print(
            "GOLD:",
            gold
        )

        print(
            "RAW:",
            raw_output
        )

        print("===================\n")

        return Score(

            value=correct,

            answer=prediction,
        )

    return score


@task
def mmlu_single():

    dataset = hf_dataset(

        path="cais/mmlu",

        name="all",

        split="test",

        sample_fields=
        mmlu_record_to_sample,
    )

    return Task(

        dataset=dataset,

        solver=
        single_agent_solver(),

        scorer=
        mmlu_scorer(),
    )


@task
def mmlu_majority_vote():

    dataset = hf_dataset(

        path="cais/mmlu",

        name="all",

        split="test",

        sample_fields=
        mmlu_record_to_sample,
    )

    return Task(

        dataset=dataset,

        solver=
        majority_vote_solver(

            agents=5,
        ),

        scorer=
        mmlu_scorer(),
    )