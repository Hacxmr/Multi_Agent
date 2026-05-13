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
# KNOWN LABEL CORRECTIONS
#
# The MMLU abstract_algebra split contains confirmed mislabeled
# questions. Each entry maps a unique substring of the question
# text to the mathematically correct answer letter.
#
# Verification for each correction is documented inline.
# =========================================================

KNOWN_LABEL_CORRECTIONS = {
    # Question: "Statement 1 | A ring homomorphism is one to one
    #            if and only if the kernel is {0}.
    #            Statement 2 | Q is an ideal in R."
    #
    # Dataset gold: D (False, True)
    # Correct answer: C (True, False)
    #
    # Statement 1 is TRUE:
    #   phi injective <=> ker(phi) = {0} holds for all ring
    #   homomorphisms by the same first isomorphism argument
    #   as for group homomorphisms (rings are groups under +).
    #
    # Statement 2 is FALSE:
    #   For Q to be an ideal in R, we need r*q in Q for all
    #   r in R, q in Q. Counterexample: sqrt(2) * 1 = sqrt(2),
    #   which is irrational, so Q is NOT an ideal in R.
    #
    "ring homomorphism is one to one if and only if the kernel": "C",
}


# =========================================================
# ROBUST MCQ EXTRACTION
# =========================================================

def extract_mcq_answer(text):
    """
    Extracts a single answer letter (A-D) from model output.

    Priority order:
      1. Explicit FINAL_ANSWER / ANSWER markers  (most reliable)
      2. Last line that is exactly one letter     (catches "Answer: C" endings)
      3. Last standalone A/B/C/D in full text     (broad fallback, least reliable)
    """

    if text is None:
        return None

    text = str(text)

    # -----------------------------------------------------
    # PRIORITY 1 — explicit final answer marker
    # -----------------------------------------------------

    priority_patterns = [
        r"FINAL_ANSWER:\s*([A-D])",
        r"ANSWER:\s*([A-D])",
        r"answer\s+is\s*[:\s]*([A-D])\b",
        r"correct\s+(?:option|answer)\s+is\s*[:\s]*([A-D])\b",
    ]

    for pattern in priority_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # -----------------------------------------------------
    # PRIORITY 2 — last line that is only a single letter
    # -----------------------------------------------------

    for line in reversed(text.splitlines()):
        line = line.strip()
        m = re.fullmatch(r"([A-D])[.\s]?", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # -----------------------------------------------------
    # PRIORITY 3 — last standalone A/B/C/D anywhere
    # -----------------------------------------------------

    option_matches = re.findall(r"\b([A-D])\b", text)
    if option_matches:
        return option_matches[-1].upper()

    return None


# =========================================================
# APPLY LABEL CORRECTIONS
# =========================================================

def apply_label_correction(question_text, original_answer):
    """
    Checks the question against KNOWN_LABEL_CORRECTIONS and
    returns the corrected answer letter if a match is found,
    or the original answer letter otherwise.

    Matching is case-insensitive substring matching on the
    question text so it is robust to minor whitespace or
    punctuation differences in the dataset.
    """

    question_lower = question_text.lower()

    for fragment, corrected_answer in KNOWN_LABEL_CORRECTIONS.items():
        if fragment.lower() in question_lower:
            print(
                f"\n[LABEL CORRECTION] Detected known mislabeled question.\n"
                f"  Fragment matched: '{fragment}'\n"
                f"  Original label:  {original_answer}\n"
                f"  Corrected label: {corrected_answer}\n"
            )
            return corrected_answer

    return original_answer


# =========================================================
# FORMAT QUESTION
# =========================================================

def format_question(record):
    """
    Formats an MMLU record as a plain text question.

    The ##mmlu## marker at the top is a hidden tag consumed by
    detect_task() in single_agent_solver.py to reliably identify
    MMLU questions without relying on visible prompt wording.
    It is never shown to the model as a meaningful instruction.
    """

    choices = record["choices"]

    question = (
        "##mmlu##\n"
        "Question:\n"
        f"{record['question']}\n\n"
        "Options:\n\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Choose the correct option."
    )

    return question.strip()


# =========================================================
# DATASET CONVERSION
# =========================================================

def mmlu_record_to_sample(record):

    answer_idx = int(record["answer"])
    raw_answer_letter = ["A", "B", "C", "D"][answer_idx]

    # Apply correction before the sample is created so the
    # corrected label flows through to the scorer unchanged.
    corrected_answer_letter = apply_label_correction(
        record["question"],
        raw_answer_letter,
    )

    return Sample(
        input=format_question(record),
        target=corrected_answer_letter,
        metadata={
            "subject": record.get("subject", ""),
            "choices": record["choices"],
            "original_answer": raw_answer_letter,
            "label_corrected": corrected_answer_letter != raw_answer_letter,
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

        # Surface label correction status in logs
        label_corrected = state.metadata.get("label_corrected", False)
        original_answer = state.metadata.get("original_answer", gold)

        correction_note = (
            f"\n[CORRECTED LABEL: dataset said {original_answer}, "
            f"using verified answer {gold}]"
            if label_corrected
            else ""
        )

        print("\n===================")
        print("QUESTION:\n", state.input_text)
        print("\nPRED:", prediction)
        print("GOLD:", gold, correction_note)
        print("CORRECT:", correct)
        print("\nRAW OUTPUT:\n")
        print(raw_output)
        print("===================\n")

        return Score(
            value=correct,
            answer=prediction,
            explanation=(
                f"Prediction: {prediction}\n\n"
                f"Gold: {gold}{correction_note}\n\n"
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