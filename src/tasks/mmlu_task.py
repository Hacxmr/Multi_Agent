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
# ROBUST MCQ EXTRACTION
# =========================================================

def extract_mcq_answer(text):

    if text is None:
        return None

    text = str(text)

    # -----------------------------------------------------
    # PRIORITY 1 — explicit final answer marker
    # -----------------------------------------------------

    patterns = [
        r"FINAL_ANSWER:\s*([A-D])",
        r"ANSWER:\s*([A-D])",
        r"answer\s+is\s*[:\s]*([A-D])\b",
        r"correct\s+(?:option|answer)\s+is\s*[:\s]*([A-D])\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # -----------------------------------------------------
    # PRIORITY 2 — last standalone letter on its own line
    # (catches "Answer: C" style endings)
    # -----------------------------------------------------

    for line in reversed(text.splitlines()):
        line = line.strip()
        m = re.fullmatch(r"([A-D])[.\s]?", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # -----------------------------------------------------
    # PRIORITY 3 — last standalone A/B/C/D in the text
    # (broad fallback; least reliable)
    # -----------------------------------------------------

    option_matches = re.findall(r"\b([A-D])\b", text)
    if option_matches:
        return option_matches[-1].upper()

    return None


# =========================================================
# FORMAT QUESTION
# =========================================================

def format_question(record):

    choices = record["choices"]

    # ##mmlu## marker is a hidden tag used by detect_task()
    # to reliably identify MMLU questions without relying
    # on the visible prompt text.
    question = f"""##mmlu##
Question:
{record["question"]}

Options:

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Choose the correct option."""

    return question.strip()


# =========================================================
# DATASET CONVERSION
# =========================================================

def mmlu_record_to_sample(record):

    answer_idx = int(record["answer"])

    answer_letter = ["A", "B", "C", "D"][answer_idx]

    return Sample(
        input=format_question(record),
        target=answer_letter,
        metadata={
            "subject": record.get("subject", ""),
            "choices": record["choices"],
        },
    )


# =========================================================
# SCORER
# =========================================================

@scorer(metrics=[accuracy()])
def mmlu_scorer():

    async def score(state, target):

        raw_output = state.output.completion

        prediction = extract_mcq_answer(raw_output)

        gold = target.text

        correct = prediction == gold

        print("\n===================")
        print("QUESTION:\n", state.input_text)
        print("\nPRED:", prediction)
        print("GOLD:", gold)
        print("\nRAW OUTPUT:\n")
        print(raw_output)
        print("===================\n")

        return Score(
            value=correct,
            answer=prediction,
            explanation=(
                f"Prediction: {prediction}\n\n"
                f"Gold: {gold}\n\n"
                f"RAW OUTPUT:\n{raw_output}"
            ),
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
        sample_fields=mmlu_record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=single_agent_solver(),
        scorer=mmlu_scorer(),
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
        sample_fields=mmlu_record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=majority_vote_solver(agents=5),
        scorer=mmlu_scorer(),
    )