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
# OPTIMAL GSM8K SETTINGS
# ============================================================

REASONING_MAX_TOKENS = 650
SYNTHESIS_MAX_TOKENS = 220

CANDIDATE_SNIPPET_CHARS = 700


# ============================================================
# DIVERSE REASONING PROMPTS
# ============================================================

REASONING_PROMPTS = [

"""
You are a careful mathematician.

Solve the problem step-by-step.

Verify arithmetic carefully.

Before giving the final answer,
recompute the result independently once.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",

"""
You are an expert math tutor.

Use explicit intermediate calculations.

Double-check arithmetic carefully.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",

"""
You are a skeptical mathematical verifier.

Actively search for mistakes
in your own reasoning.

Recompute the final result.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",

"""
You are a competition math solver.

Solve efficiently but carefully.

Verify units and arithmetic.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",

"""
You are a rigorous accountant.

Track all quantities carefully.

Check every arithmetic operation.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",
]


# ============================================================
# SYNTHESIS JUDGE
# ONLY USED FOR TRUE TIES
# ============================================================

SYNTHESIS_SYSTEM = """
You are a mathematical judge.

You will receive multiple candidate solutions.

IMPORTANT:
1. Verify calculations independently.
2. Do NOT trust majority blindly.
3. Look carefully for arithmetic mistakes.
4. Prefer logically correct reasoning.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
"""


# ============================================================
# NUMBER NORMALIZATION
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
    text = text.strip()

    # --------------------------------------------------------
    # STRICT EXTRACTION ONLY
    # --------------------------------------------------------

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",

        r"FINAL[_ ]ANSWER:\s*([-+]?\d*\.?\d+)",

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
    # LAST NUMERIC LINE ONLY
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

    cleaned = strip_think_blocks(text)

    if len(cleaned) <= max_chars:

        return cleaned

    return cleaned[-max_chars:]


# ============================================================
# EXTRACTION FALLBACK
# ============================================================

async def llm_extract_answer(

    model,

    text,
):

    prompt = f"""
Extract the FINAL numeric answer.

Return ONLY the final number.

Solution:
{text}
"""

    response = await model.generate(

        [

            ChatMessageUser(
                content=prompt
            ),
        ],

        config=GenerateConfig(

            temperature=0.0,

            max_tokens=12,
        ),
    )

    extracted = response.completion.strip()

    return extract_answer(
        extracted
    )


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
# MAIN SOLVER
# ============================================================

@solver
def debate_solver(

    agents=5,

    rounds=1,  # backward compatibility

    use_synthesis_judge=True,

    base_temperature=0.20,

    temperature_spread=0.15,
):

    """
    Optimized GSM8K Multi-Agent Debate Solver

    Key design principles:
    - Diverse prompts
    - Self-consistency voting
    - Judge ONLY on ties
    - Minimal complexity
    - Stable extraction
    """

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        fallback_extractions = 0

        candidate_outputs = []

        candidate_answers = []

        # ====================================================
        # INDEPENDENT REASONING
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

            # ------------------------------------------------
            # EXTRACTION FALLBACK
            # ------------------------------------------------

            if answer is None:

                fallback_extractions += 1

                answer = await llm_extract_answer(

                    model,

                    completion,
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
        # JUDGE ONLY ON TRUE TIES
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
{trim_candidate(out)}
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
PROBLEM:
{problem}

CANDIDATE SOLUTIONS:
{candidate_summary}

Determine the MOST correct answer.

Carefully verify arithmetic and logic.
"""

            synthesis_text, judge_answer = await run_agent(

                model=model,

                system_prompt=SYNTHESIS_SYSTEM,

                user_prompt=judge_prompt,

                temperature=0.0,

                max_tokens=SYNTHESIS_MAX_TOKENS,
            )

            # ------------------------------------------------
            # FALLBACK EXTRACTION
            # ------------------------------------------------

            if judge_answer is None:

                fallback_extractions += 1

                judge_answer = await llm_extract_answer(

                    model,

                    synthesis_text,
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

        # ====================================================
        # METADATA
        # ====================================================

        state.metadata[
            "candidate_answers"
        ] = candidate_answers

        state.metadata[
            "majority_vote"
        ] = voted_answer

        state.metadata[
            "final_answer"
        ] = final_answer

        state.metadata[
            "judge_used"
        ] = judge_used

        state.metadata[
            "fallback_extractions"
        ] = fallback_extractions

        state.metadata[
            "agents"
        ] = agents

        state.metadata[
            "rounds"
        ] = rounds

        return state

    return solve