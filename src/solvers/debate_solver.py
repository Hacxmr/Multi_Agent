import re
from collections import Counter
from typing import Optional

from inspect_ai.solver import solver, Generate

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)

# ============================================================
# SETTINGS
# ============================================================

REASONING_MAX_TOKENS = 800

# ============================================================
# SIMPLE PROMPTS
# ============================================================

REASONING_PROMPTS = [

"""
Solve the math problem step-by-step.

Give the final numeric answer clearly.
""",

"""
Carefully solve the math problem.

Show calculations clearly.
""",

"""
Solve carefully and avoid arithmetic mistakes.
""",

"""


Compute the answer step-by-step.
""",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_number(value):

    try:

        value = str(value)

        value = value.replace(",", "")
        value = value.replace("$", "")
        value = value.strip()

        value = float(value)

        if value.is_integer():

            return str(int(value))

        return str(round(value, 6))

    except (ValueError, TypeError):

        return None


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(text: str) -> Optional[str]:

    if text is None:
        return None

    text = str(text)

    text = text.replace(",", "")
    text = text.replace("$", "")

    # --------------------------------------------------------
    # STRUCTURED PATTERNS
    # --------------------------------------------------------

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"answer is\s*([-+]?\d*\.?\d+)",

        r"final answer.*?([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",
    ]

    for pattern in patterns:

        matches = re.findall(

            pattern,

            text,

            re.IGNORECASE,
        )

        if matches:

            return normalize_number(
                matches[-1]
            )

    # --------------------------------------------------------
    # LAST NUMERIC LINE
    # --------------------------------------------------------

    lines = text.splitlines()

    for line in reversed(lines):

        line = line.strip()

        match = re.fullmatch(

            r"[-+]?\d*\.?\d+",

            line,
        )

        if match:

            return normalize_number(
                match.group(0)
            )

    # --------------------------------------------------------
    # FINAL NUMBER FALLBACK
    # --------------------------------------------------------

    numbers = re.findall(

        r"[-+]?\d*\.?\d+",

        text,
    )

    if numbers:

        return normalize_number(
            numbers[-1]
        )

    return None


# ============================================================
# CLEANING
# ============================================================

def strip_think_blocks(text):

    if text is None:
        return ""

    text = re.sub(

        r"<think>.*?</think>",

        "",

        text,

        flags=re.DOTALL,
    )

    return text.strip()


# ============================================================
# MODEL CALL
# ============================================================

async def run_agent(

    model,

    system_prompt,

    user_prompt,

    temperature,

    max_tokens,
):

    response = await model.generate(

        [

            ChatMessageSystem(
                content=system_prompt
            ),

            ChatMessageUser(
                content=user_prompt
            ),
        ],

        config=GenerateConfig(

            temperature=temperature,

            top_p=0.95,

            max_tokens=max_tokens,
        ),
    )

    completion = strip_think_blocks(
        response.completion
    )

    answer = extract_answer(
        completion
    )

    return completion, answer


# ============================================================
# MAIN SOLVER
# ============================================================

@solver
def debate_solver(

    agents=7,

    rounds=1,  # backward compatibility

    base_temperature=0.15,

    temperature_spread=0.08,
):

    """
    Final optimized GSM8K solver.

    Key observations:
    - Majority voting helps
    - Judge hurts accuracy
    - Simpler architecture works better
    - More agents help more than debate
    """

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        candidate_outputs = []

        candidate_answers = []

        # ====================================================
        # INDEPENDENT AGENTS
        # ====================================================

        for agent_id in range(agents):

            temperature = (

                base_temperature +

                (
                    agent_id *
                    temperature_spread
                )
            )

            system_prompt = REASONING_PROMPTS[
                agent_id % len(REASONING_PROMPTS)
            ]

            completion, answer = await run_agent(

                model=model,

                system_prompt=system_prompt,

                user_prompt=problem,

                temperature=temperature,

                max_tokens=REASONING_MAX_TOKENS,
            )

            candidate_outputs.append(
                completion
            )

            candidate_answers.append(
                answer
            )

        # ====================================================
        # SELF-CONSISTENCY VOTING
        # ====================================================

        valid_answers = [

            a for a in candidate_answers
            if a is not None
        ]

        answer_counts = Counter(
            valid_answers
        )

        most_common = answer_counts.most_common()

        voted_answer = "UNKNOWN"

        if len(most_common) > 0:

            voted_answer = most_common[0][0]

        final_answer = voted_answer

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        answer_summary = "\n".join(

            [

                f"Agent {i+1}: {ans}"

                for i, ans in enumerate(
                    candidate_answers
                )
            ]
        )

        state.output.completion = f"""
Independent Agent Answers:

{answer_summary}

Majority Vote:
{voted_answer}

#### {final_answer}
"""

        # ====================================================
        # METADATA
        # ====================================================

        state.metadata["candidate_answers"] = candidate_answers
        state.metadata["majority_vote"] = voted_answer
        state.metadata["final_answer"] = final_answer
        state.metadata["agents"] = agents
        state.metadata["rounds"] = rounds

        return state

    return solve