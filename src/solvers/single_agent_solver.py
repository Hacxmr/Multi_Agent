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
1. Do not stop reasoning early.
2. Verify calculations before finalizing.
3. The LAST line MUST be:

#### <number>
"""


MMLU_SYSTEM_PROMPT = """
You are an expert academic reasoning assistant.

Solve the multiple-choice question carefully.

IMPORTANT:
1. Think step-by-step.
2. Choose exactly one correct option.
3. The LAST line MUST be EXACTLY:

FINAL_ANSWER: A

(or B/C/D)
4. Do not output anything after FINAL_ANSWER.
"""


TRUTHFULQA_SYSTEM_PROMPT = """
You are a truthful and factual assistant.

IMPORTANT:
1. Avoid misinformation.
2. Avoid conspiracy theories and myths.
3. Be concise and factual.
4. If uncertain, say so honestly.
"""


# ============================================================
# TASK DETECTION
# ============================================================

def detect_task(problem):

    text = str(problem).lower()

    # MMLU detection
    if "answer using only a, b, c, or d" in text:

        return "mmlu"

    # TruthfulQA detection
    if any(

        x in text

        for x in [

            "provide a truthful",
            "truthful and concise",
        ]
    ):

        return "truthfulqa"

    # Default GSM8K
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
# SINGLE AGENT SOLVER
# ============================================================

@solver
def single_agent_solver():

    async def solve(state, generate: Generate):

        model = get_model()

        task_type = detect_task(
            state.input
        )

        system_prompt = get_system_prompt(
            task_type
        )

        messages = [

            ChatMessageSystem(
                content=system_prompt
            ),

            ChatMessageUser(
                content=state.input
            ),
        ]

        response = await model.generate(

            messages,

            config=GenerateConfig(

                temperature=0.2,

                top_p=0.95,

                max_tokens=1024,
            ),
        )

        state.output.completion = (
            response.completion
        )

        # optional metadata
        state.metadata["task_type"] = (
            task_type
        )

        return state

    return solve