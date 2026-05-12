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
SYNTHESIS_MAX_TOKENS = 300

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

Show the calculations clearly.
""",

"""
Solve carefully and avoid arithmetic mistakes.
""",
]


SYNTHESIS_SYSTEM = """
You are a mathematical judge.

Choose the most correct answer.
"""


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
    # #### FORMAT
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

    agents=5,

    rounds=1,

    use_synthesis_judge=True,

    base_temperature=0.2,

    temperature_spread=0.1,
):

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        candidate_outputs = []

        candidate_answers = []

        # ====================================================
        # AGENT REASONING
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
        # MAJORITY VOTE
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

        tie = False

        if len(most_common) > 0:

            voted_answer = most_common[0][0]

            if len(most_common) > 1:

                if (
                    most_common[0][1]
                    ==
                    most_common[1][1]
                ):

                    tie = True

        final_answer = voted_answer

        synthesis_text = ""

        judge_used = False

        # ====================================================
        # TIE-BREAK JUDGE
        # ====================================================

        if use_synthesis_judge and tie:

            judge_used = True

            candidate_summary = "\n\n".join(

                [

                    f"""
Agent {i+1}

Answer:
{ans}

Solution:
{out}
"""

                    for i, (

                        out,
                        ans

                    ) in enumerate(

                        zip(
                            candidate_outputs,
                            candidate_answers,
                        )
                    )
                ]
            )

            judge_prompt = f"""
Problem:
{problem}

Candidate Solutions:
{candidate_summary}

Choose the most correct answer.
"""

            synthesis_text, judge_answer = await run_agent(

                model=model,

                system_prompt=SYNTHESIS_SYSTEM,

                user_prompt=judge_prompt,

                temperature=0.0,

                max_tokens=SYNTHESIS_MAX_TOKENS,
            )

            if judge_answer is not None:

                final_answer = judge_answer

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

Judge Used:
{judge_used}

Judge Decision:

{synthesis_text}

#### {final_answer}
"""

        return state

    return solve