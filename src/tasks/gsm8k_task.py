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

from src.solvers.debate_solver import (
    debate_solver,
)


def normalize_number(text):

    if text is None:
        return None

    text = str(text)

    text = text.replace(",", "")
    text = text.strip()

    try:

        value = float(text)

        if value.is_integer():
            return str(int(value))

        return str(round(value, 6))

    except:

        return text.strip()


def extract_answer(text):

    if text is None:
        return None

    text = str(text)

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"FINAL_ANSWER:\s*([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",

        r"answer is\s*([-+]?\d*\.?\d+)",

        r"therefore.*?([-+]?\d*\.?\d+)",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if matches:

            return normalize_number(
                matches[-1]
            )

    lines = text.strip().split("\n")

    for line in reversed(lines):

        numbers = re.findall(
            r"[-+]?\d*\.?\d+",
            line,
        )

        if numbers:

            return normalize_number(
                numbers[-1]
            )

    return None


def gsm8k_record_to_sample(record):

    gold = extract_answer(
        record["answer"]
    )

    return Sample(

        input=record["question"],

        target=gold,

        metadata={

            "raw_answer": record["answer"],
        },
    )


@scorer(metrics=[accuracy()])
def gsm8k_scorer():

    async def score(state, target):

        raw_output = state.output.completion

        prediction = extract_answer(
            raw_output
        )

        gold = target.text

        correct = prediction == gold

        print("\n===================")
        print("PRED:", prediction)
        print("GOLD:", gold)
        print("RAW:", raw_output)
        print("===================\n")

        return Score(

            value=correct,

            answer=prediction,

            explanation=f"""
Prediction: {prediction}

Gold: {gold}

RAW OUTPUT:
{raw_output}
""",
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