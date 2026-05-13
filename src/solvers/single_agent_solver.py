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

Solve the multiple-choice question carefully, thinking step-by-step.

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
    # MMLU — match the marker injected by format_question
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
    # MMLU — slightly more tokens for chain-of-thought
    # --------------------------------------------------------

    if task_type == "mmlu":
        return GenerateConfig(
            temperature=0.1,
            top_p=0.95,
            max_tokens=2048,
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

        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        state.output.completion = response.completion.strip()

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        state.metadata["task_type"] = task_type

        return state

    return solve