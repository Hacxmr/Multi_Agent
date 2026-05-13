from inspect_ai.solver import (
    solver,
    Generate,
)

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)


# ============================================================
# PROMPTS
# ============================================================

GSM8K_SYSTEM_PROMPT = """
You are an expert mathematical reasoning assistant.

Solve problems carefully step-by-step.

IMPORTANT:
The LAST line MUST be:

#### <number>
"""


MMLU_SYSTEM_PROMPT = """
You are an expert academic reasoning assistant.

Reason through the question step-by-step in under 300 words.

Rules:
- Once you reach a conclusion supported by a counterexample, do NOT reverse it based on a single confirming case. Counterexamples are decisive.
- Do not second-guess a correct conclusion at the end.

IMPORTANT:
The LAST line of your response MUST be EXACTLY in this format with no extra text:

FINAL_ANSWER: A

(Replace A with whichever letter — A, B, C, or D — is correct.)
"""


TRUTHFULQA_SYSTEM_PROMPT = """
You are a truthful and factual assistant.

IMPORTANT:
1. Avoid misinformation.
2. Be concise and factual.
3. If uncertain, say so honestly.
"""


# ============================================================
# TASK DETECTION
# ============================================================

def detect_task(problem):

    text = str(problem).lower()

    # --------------------------------------------------------
    # MMLU — match the hidden marker injected by format_question
    # --------------------------------------------------------

    if "##mmlu##" in text:
        return "mmlu"

    # --------------------------------------------------------
    # TRUTHFULQA
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "truthful",
            "misinformation",
            "myth",
            "conspiracy",
        ]
    ):
        return "truthfulqa"

    # --------------------------------------------------------
    # GSM8K (default)
    # --------------------------------------------------------

    return "gsm8k"


# ============================================================
# PROMPT SELECTION
# ============================================================

def get_system_prompt(task_type):

    if task_type == "gsm8k":
        return GSM8K_SYSTEM_PROMPT

    elif task_type == "mmlu":
        return MMLU_SYSTEM_PROMPT

    elif task_type == "truthfulqa":
        return TRUTHFULQA_SYSTEM_PROMPT

    return GSM8K_SYSTEM_PROMPT


# ============================================================
# CONFIG SELECTION
# ============================================================

def get_generation_config(task_type):

    # --------------------------------------------------------
    # MMLU — 3000 tokens to avoid mid-reasoning truncation
    # --------------------------------------------------------

    if task_type == "mmlu":
        return GenerateConfig(
            temperature=0.1,
            top_p=0.95,
            max_tokens=3000,
        )

    # --------------------------------------------------------
    # TRUTHFULQA
    # --------------------------------------------------------

    elif task_type == "truthfulqa":
        return GenerateConfig(
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
        )

    # --------------------------------------------------------
    # GSM8K
    # --------------------------------------------------------

    return GenerateConfig(
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
    )


# ============================================================
# FALLBACK EXTRACTION via a second model call
# Used when the primary output is truncated or unparseable.
# ============================================================

async def extract_with_fallback(raw_output, original_question, model):
    """
    Makes a minimal follow-up call to recover an answer letter
    when the primary output did not contain a parseable FINAL_ANSWER.
    Takes only the last 600 characters of the output to keep the
    extraction call cheap.
    """

    tail = raw_output[-600:] if raw_output else ""

    extraction_prompt = (
        "The following is a partial or incomplete response to a "
        "multiple-choice question. Based solely on the reasoning "
        "shown, which answer letter (A, B, C, or D) was the "
        "response converging toward?\n\n"
        f"Question:\n{original_question}\n\n"
        f"Partial response (end of text):\n{tail}\n\n"
        "Reply with ONLY a single uppercase letter: A, B, C, or D."
    )

    response = await model.generate(
        [ChatMessageUser(content=extraction_prompt)],
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=5,
        ),
    )

    return response.completion.strip()


# ============================================================
# SINGLE AGENT SOLVER
# ============================================================

@solver
def single_agent_solver():

    async def solve(state, generate: Generate):

        model = get_model()

        # ----------------------------------------------------
        # DETECT TASK
        # ----------------------------------------------------

        task_type = detect_task(state.input)

        # ----------------------------------------------------
        # SELECT PROMPT & CONFIG
        # ----------------------------------------------------

        system_prompt = get_system_prompt(task_type)
        generation_config = get_generation_config(task_type)

        # ----------------------------------------------------
        # BUILD MESSAGES
        # ----------------------------------------------------

        messages = [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=state.input),
        ]

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        response = await model.generate(
            messages,
            config=generation_config,
        )

        raw_output = response.completion.strip()

        # ----------------------------------------------------
        # FALLBACK for MMLU: if no parseable answer found,
        # make a second cheap call to recover the answer letter.
        # Imported here to avoid a circular import at module load.
        # ----------------------------------------------------

        if task_type == "mmlu":

            from src.tasks.mmlu_task import extract_mcq_answer

            tentative = extract_mcq_answer(raw_output)

            if tentative is None:

                fallback_raw = await extract_with_fallback(
                    raw_output,
                    state.input,
                    model,
                )

                # Append a clean FINAL_ANSWER line so the scorer
                # can extract it with the normal regex pattern.
                raw_output = (
                    raw_output
                    + f"\n\nFINAL_ANSWER: {fallback_raw}"
                )

        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        state.output.completion = raw_output

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        state.metadata["task_type"] = task_type

        return state

    return solve