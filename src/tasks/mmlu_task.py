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


# =========================================================
# ROBUST ANSWER EXTRACTION
# =========================================================

def extract_mcq_answer(text):

    if text is None:
        return None

    text = str(text)

    # -----------------------------------------------------
    # STRICT FINAL ANSWER PATTERNS
    # -----------------------------------------------------

    patterns = [

        r"FINAL_ANSWER:\s*([A-D])",

        r"ANSWER:\s*([A-D])",

        r"answer\s+is\s*([A-D])",

        r"correct\s+answer\s+is\s*([A-D])",

        r"option\s*([A-D])",

        r"^\s*([A-D])\s*$",
    ]

    for pattern in patterns:

        matches = re.findall(

            pattern,

            text,

            re.IGNORECASE | re.MULTILINE,
        )

        if matches:

            return matches[-1].upper()

    # -----------------------------------------------------
    # FALLBACK:
    # LAST STANDALONE LETTER
    # -----------------------------------------------------

    matches = re.findall(

        r"\b([A-D])\b",

        text,
    )

    if matches:

        return matches[-1].upper()

    return None


# =========================================================
# FORMAT QUESTION
# =========================================================

def format_question(record):

    choices = record["choices"]

    question = f"""
Question:
{record["question"]}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

IMPORTANT:
Return ONLY the final option.

The LAST line MUST be EXACTLY:

FINAL_ANSWER: A

(or B/C/D)
"""

    return question.strip()


# =========================================================
# DATASET CONVERSION
# =========================================================

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

        metadata={

            "subject": record.get(
                "subject",
                ""
            ),

            "choices": record[
                "choices"
            ],
        },
    )


# =========================================================
# SCORER
# =========================================================

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
            "QUESTION:\n",
            state.input_text
        )

        print("\nPRED:", prediction)

        print("GOLD:", gold)

        print("\nRAW OUTPUT:\n")
        print(raw_output)

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


# =========================================================
# SINGLE AGENT TASK
# =========================================================

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


# =========================================================
# MAJORITY VOTE TASK
# =========================================================

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

        solver=majority_vote_solver(

            agents=5,
        ),

        scorer=
        mmlu_scorer(),
    )