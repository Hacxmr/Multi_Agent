import re

from rapidfuzz import fuzz

from sentence_transformers import (
    SentenceTransformer,
    util,
)

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


semantic_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_candidate_answer(text):

    if text is None:
        return ""

    text = clean_text(text)

    patterns = [

        r"####\s*(.*)",

        r"final_answer:\s*(.*)",

        r"answer is\s*(.*)",

        r"therefore[,]?\s*(.*)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            return match.group(1).strip()

    lines = text.split("\n")

    if lines:

        return lines[-1].strip()

    return text.strip()


def semantic_match(prediction, gold):

    pred_clean = clean_text(prediction)

    gold_clean = clean_text(gold)

    if pred_clean == gold_clean:
        return True

    fuzzy_score = fuzz.ratio(
        pred_clean,
        gold_clean,
    )

    if fuzzy_score > 90:
        return True

    emb1 = semantic_model.encode(
        pred_clean,
        convert_to_tensor=True,
    )

    emb2 = semantic_model.encode(
        gold_clean,
        convert_to_tensor=True,
    )

    similarity = util.cos_sim(
        emb1,
        emb2,
    ).item()

    return similarity > 0.92


def gsm8k_record_to_sample(record):

    return Sample(

        input=record["question"],

        target=record["answer"],

        metadata={

            "solution": record["answer"],
        },
    )


@scorer(metrics=[accuracy()])
def gsm8k_scorer():

    async def score(state, target):

        prediction = extract_candidate_answer(
            state.output.completion
        )

        gold = extract_candidate_answer(
            target.text
        )

        correct = semantic_match(
            prediction,
            gold,
        )

        return Score(

            value=correct,

            answer=prediction,

            explanation=f"""
Prediction:
{prediction}

Gold:
{gold}

RAW OUTPUT:
{state.output.completion}
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