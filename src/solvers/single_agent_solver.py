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

Solve the problem carefully step-by-step.

IMPORTANT:
1. Verify calculations carefully.
2. The LAST line MUST be EXACTLY:

#### <number>

3. Do not output anything after the final answer.
"""


MMLU_SYSTEM_PROMPT = """
You are an expert academic multiple-choice assistant.

IMPORTANT RULES:
1. Think silently.
2. Do NOT explain your reasoning.
3. Choose EXACTLY one option.
4. Output ONLY the final answer.
5. The response MUST be EXACTLY:

FINAL_ANSWER: A

(or B/C/D)

6. Do not output any extra text.
"""


TRUTHFULQA_SYSTEM_PROMPT = """
You are a truthful and factual assistant.

IMPORTANT:
1. Be concise and accurate.
2. Avoid speculation and misinformation.
3. If uncertain, say so honestly.
4. Keep responses short and factual.
"""


# ============================================================
# TASK DETECTION
# ============================================================

def detect_task(problem):

    text = str(problem).lower()

    # --------------------------------------------------------
    # MMLU DETECTION
    # --------------------------------------------------------

    if (

        "answer using only a, b, c, or d"

        in text

        or "final_answer:" in text

    ):

        return "mmlu"

    # --------------------------------------------------------
    # TRUTHFULQA DETECTION
    # --------------------------------------------------------

    if any(

        x in text

        for x in [

            "truthful",

            "misinformation",

            "conspiracy",

            "myth",
        ]
    ):

        return "truthfulqa"

    # --------------------------------------------------------
    # DEFAULT GSM8K
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
    # MMLU
    # --------------------------------------------------------

    if task_type == "mmlu":

        return GenerateConfig(

            temperature=0.0,

            top_p=1.0,

            max_tokens=10,

            stop=[

                "\n\n",

                "</think>",
            ],
        )

    # --------------------------------------------------------
    # TRUTHFULQA
    # --------------------------------------------------------

    elif task_type == "truthfulqa":

        return GenerateConfig(

            temperature=0.2,

            top_p=0.9,

            max_tokens=128,
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

        task_type = detect_task(
            state.input
        )

        # ----------------------------------------------------
        # SELECT PROMPT
        # ----------------------------------------------------

        system_prompt = get_system_prompt(
            task_type
        )

        # ----------------------------------------------------
        # SELECT CONFIG
        # ----------------------------------------------------

        generation_config = (
            get_generation_config(
                task_type
            )
        )

        # ----------------------------------------------------
        # BUILD MESSAGES
        # ----------------------------------------------------

        messages = [

            ChatMessageSystem(
                content=system_prompt
            ),

            ChatMessageUser(
                content=state.input
            ),
        ]

        # ----------------------------------------------------
        # GENERATE RESPONSE
        # ----------------------------------------------------

        response = await model.generate(

            messages,

            config=generation_config,
        )

        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        state.output.completion = (
            response.completion.strip()
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        state.metadata["task_type"] = (
            task_type
        )

        return state

    return solve
