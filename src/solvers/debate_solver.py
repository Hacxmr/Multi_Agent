import re
from collections import Counter
from typing import Optional

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
# TOKEN BUDGETS
# ============================================================

REASONING_MAX_TOKENS = 1200
CRITIQUE_MAX_TOKENS = 900
SYNTHESIS_MAX_TOKENS = 600

CANDIDATE_SNIPPET_CHARS = 1200


# ============================================================
# SYSTEM PROMPTS
# ============================================================

REASONING_SYSTEM = """
You are an expert mathematical reasoning assistant.

Solve carefully step-by-step.

IMPORTANT:
- Verify arithmetic carefully.
- Keep reasoning concise.
- The FINAL line MUST contain ONLY the numeric answer.

Example:
42
"""


CRITIQUE_SYSTEM = """
You are a mathematical critic.

Review other solutions carefully.

IMPORTANT:
- Detect arithmetic mistakes.
- Revise if another answer is better.
- Keep reasoning concise.
- The FINAL line MUST contain ONLY the numeric answer.

Example:
42
"""


SYNTHESIS_SYSTEM = """
You are a mathematical judge.

Choose the most correct answer.

IMPORTANT:
- Verify calculations independently.
- Do not blindly trust majority.
- Keep reasoning concise.
- The FINAL line MUST contain ONLY the numeric answer.

Example:
42
"""


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def normalize_number(value):

    try:

        value = float(value)

        if value.is_integer():
            return str(int(value))

        return str(value)

    except:
        return None


def extract_answer(text: str) -> Optional[str]:

    if text is None:
        return None

    text = text.replace(",", "").strip()

    # --------------------------------------------------------
    # STRICT PATTERNS
    # --------------------------------------------------------

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",

        r"FINAL ANSWER:\s*([-+]?\d*\.?\d+)",

        r"answer is\s*([-+]?\d*\.?\d+)",

        r"^\s*([-+]?\d*\.?\d+)\s*$",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE | re.MULTILINE,
        )

        if matches:

            return normalize_number(
                matches[-1]
            )

    # --------------------------------------------------------
    # FALLBACK:
    # LAST NUMBER IN ENTIRE OUTPUT
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


def trim_candidate(

    text,

    max_chars=CANDIDATE_SNIPPET_CHARS,
):

    cleaned = strip_think_blocks(
        text
    )

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[-max_chars:]


def majority_vote(answers):

    valid_answers = [

        a for a in answers
        if a is not None
    ]

    if not valid_answers:
        return "UNKNOWN"

    return Counter(
        valid_answers
    ).most_common(1)[0][0]


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

    completion = response.completion

    completion = strip_think_blocks(
        completion
    )

    answer = extract_answer(
        completion
    )

    return completion, answer


# ============================================================
# MULTI AGENT DEBATE
# ============================================================

@solver
def debate_solver(

    agents=3,

    rounds=2,

    use_synthesis_judge=True,

    base_temperature=0.15,

    temperature_spread=0.05,
):

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        # ====================================================
        # ROUND 1
        # ====================================================

        round_outputs = []

        round_answers = []

        for agent_id in range(agents):

            temperature = (

                base_temperature +

                (
                    agent_id *

                    temperature_spread
                )
            )

            completion, answer = await run_agent(

                model=model,

                system_prompt=REASONING_SYSTEM,

                user_prompt=problem,

                temperature=temperature,

                max_tokens=REASONING_MAX_TOKENS,
            )

            round_outputs.append(
                completion
            )

            round_answers.append(
                answer
            )

        # ====================================================
        # DEBATE ROUNDS
        # ====================================================

        for _ in range(rounds):

            new_outputs = []

            new_answers = []

            snippets = "\n\n".join(

                [

                    f"""
Agent {i+1}

Answer:
{ans}

Solution:
{trim_candidate(out)}
"""

                    for i, (

                        out,
                        ans

                    ) in enumerate(

                        zip(
                            round_outputs,
                            round_answers,
                        )
                    )
                ]
            )

            for agent_id in range(agents):

                temperature = (

                    base_temperature +

                    (
                        agent_id *

                        temperature_spread
                    )
                )

                critique_prompt = f"""
PROBLEM:
{problem}

OTHER AGENT SOLUTIONS:
{snippets}

Review carefully.

Revise ONLY if another solution is better.
"""

                completion, answer = await run_agent(

                    model=model,

                    system_prompt=CRITIQUE_SYSTEM,

                    user_prompt=critique_prompt,

                    temperature=temperature,

                    max_tokens=CRITIQUE_MAX_TOKENS,
                )

                new_outputs.append(
                    completion
                )

                new_answers.append(
                    answer
                )

            round_outputs = new_outputs

            round_answers = new_answers

        # ====================================================
        # SYNTHESIS
        # ====================================================

        final_answer = None

        synthesis_text = ""

        if use_synthesis_judge:

            answer_roster = "\n".join(

                [

                    f"""
Agent {i+1}: {ans}
"""

                    for i, ans in enumerate(
                        round_answers
                    )
                ]
            )

            judge_prompt = f"""
PROBLEM:
{problem}

CANDIDATE ANSWERS:
{answer_roster}

Choose the most correct answer.
"""

            synthesis_text, final_answer = await run_agent(

                model=model,

                system_prompt=SYNTHESIS_SYSTEM,

                user_prompt=judge_prompt,

                temperature=0.0,

                max_tokens=SYNTHESIS_MAX_TOKENS,
            )

        # ====================================================
        # FALLBACK
        # ====================================================

        if final_answer is None:

            final_answer = majority_vote(
                round_answers
            )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        answer_summary = "\n".join(

            [

                f"Agent {i+1}: {ans}"

                for i, ans in enumerate(
                    round_answers
                )
            ]
        )

        state.output.completion = f"""
Post-debate answers:

{answer_summary}

Judge:

{synthesis_text}

Final Answer:
{final_answer}
"""

        # ====================================================
        # METADATA
        # ====================================================

        state.metadata[
            "round_answers"
        ] = round_answers

        state.metadata[
            "final_answer"
        ] = final_answer

        return state

    return solve