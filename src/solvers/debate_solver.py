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
# CONFIGURATION
# ============================================================

REASONING_MAX_TOKENS = 700
VERIFICATION_MAX_TOKENS = 180
SYNTHESIS_MAX_TOKENS = 300

CANDIDATE_SNIPPET_CHARS = 900


# ============================================================
# DYNAMIC REASONING PROMPTS
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

Check each reasoning step carefully.

Before giving the final answer,
double-check all arithmetic.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",

"""
You are a skeptical mathematical verifier.

Actively search for mistakes in your own reasoning.

Recompute the final result independently.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",

"""
You are a competition math solver.

Solve efficiently but carefully.

Verify units, arithmetic, and assumptions.

Before answering, recompute the final result.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",

"""
You are a rigorous accountant.

Track all quantities carefully.

Check every arithmetic operation.

Before answering, independently verify the result.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
""",
]


SYNTHESIS_SYSTEM = """
You are a mathematical judge.

You will receive multiple candidate solutions.

IMPORTANT:
1. Verify calculations independently.
2. Do NOT trust the majority blindly.
3. Look carefully for arithmetic mistakes.
4. Prefer logically correct reasoning.
5. Recompute the final result yourself.

The FINAL line MUST contain ONLY the numeric answer.

Example:
42
"""


VERIFICATION_SYSTEM = """
You are a mathematical verifier.

Check the provided solution carefully.

Look for:
- arithmetic mistakes
- incorrect assumptions
- logical inconsistencies
- missed steps

If incorrect, fix the solution.

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

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",

        r"FINAL[_ ]ANSWER:\s*([-+]?\d*\.?\d+)",

        r"answer is\s*([-+]?\d*\.?\d+)",

        r"therefore.*?([-+]?\d*\.?\d+)",

        r"^\s*([-+]?\d*\.?\d+)\s*$",
    ]

    # --------------------------------------------------------
    # STRICT PATTERNS
    # --------------------------------------------------------

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
    # FINAL FALLBACK
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

    cleaned = strip_think_blocks(text)

    if len(cleaned) <= max_chars:

        return cleaned

    return cleaned[-max_chars:]


# ============================================================
# LLM EXTRACTION FALLBACK
# ============================================================

async def llm_extract_answer(

    model,

    text,
):

    prompt = f"""
Extract the FINAL numeric answer from the solution below.

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

            max_tokens=20,
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
# VERIFICATION PASS
# ============================================================

async def verify_solution(

    model,

    problem,

    solution,
):

    verification_prompt = f"""
PROBLEM:
{problem}

SOLUTION:
{solution}

Verify the solution carefully.

If incorrect, fix it.

The FINAL line MUST contain ONLY the numeric answer.
"""

    completion, answer = await run_agent(

        model=model,

        system_prompt=VERIFICATION_SYSTEM,

        user_prompt=verification_prompt,

        temperature=0.0,

        max_tokens=VERIFICATION_MAX_TOKENS,
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

    use_verification=True,

    dynamic_judge=True,

    base_temperature=0.20,

    temperature_spread=0.15,
):

    """
    Dynamic GSM8K Debate Solver

    Features:
    - Diverse reasoning prompts
    - Verification pass
    - Dynamic judge activation
    - Self-consistency voting
    - Backward compatibility
    """

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        fallback_extractions = 0

        verification_repairs = 0

        candidate_outputs = []

        candidate_answers = []

        verification_logs = []

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

            # ------------------------------------------------
            # VERIFICATION PASS
            # ------------------------------------------------

            verification_text = ""

            if use_verification:

                verification_text, verified_answer = await verify_solution(

                    model,

                    problem,

                    completion,
                )

                if verified_answer is None:

                    fallback_extractions += 1

                    verified_answer = await llm_extract_answer(

                        model,

                        verification_text,
                    )

                # --------------------------------------------
                # Use repaired answer if verifier changed it
                # --------------------------------------------

                if (
                    verified_answer is not None
                    and
                    verified_answer != answer
                ):

                    verification_repairs += 1

                    answer = verified_answer

            candidate_outputs.append(
                completion
            )

            candidate_answers.append(
                answer
            )

            verification_logs.append(
                verification_text
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

        synthesis_text = ""

        judge_used = False

        unique_answers = len(
            set(valid_answers)
        )

        majority_strength = 0

        if len(most_common) > 0:

            majority_strength = (

                most_common[0][1]
                / agents
            )

        # ====================================================
        # DYNAMIC JUDGE ACTIVATION
        # ====================================================

        run_judge = False

        if use_synthesis_judge:

            # --------------------------------------------
            # Any disagreement
            # --------------------------------------------

            if unique_answers > 1:

                run_judge = True

            # --------------------------------------------
            # Weak majority
            # --------------------------------------------

            if majority_strength < 0.60:

                run_judge = True

        # ====================================================
        # SYNTHESIS JUDGE
        # ====================================================

        if run_judge:

            judge_used = True

            candidate_summary = "\n\n".join(

                [

                    f"""
Agent {i+1}

Answer:
{ans}

Solution:
{trim_candidate(out)}

Verification:
{trim_candidate(ver)}
"""

                    for i, (

                        out,
                        ans,
                        ver

                    ) in enumerate(

                        zip(
                            candidate_outputs,
                            candidate_answers,
                            verification_logs,
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

Do NOT trust the majority blindly.

Carefully verify arithmetic and logic.

Recompute the answer independently.
"""

            synthesis_text, judge_answer = await run_agent(

                model=model,

                system_prompt=SYNTHESIS_SYSTEM,

                user_prompt=judge_prompt,

                temperature=0.0,

                max_tokens=SYNTHESIS_MAX_TOKENS,
            )

            # ------------------------------------------------
            # Judge extraction fallback
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
            "unique_answers"
        ] = unique_answers

        state.metadata[
            "majority_strength"
        ] = majority_strength

        state.metadata[
            "fallback_extractions"
        ] = fallback_extractions

        state.metadata[
            "verification_repairs"
        ] = verification_repairs

        state.metadata[
            "agents"
        ] = agents

        state.metadata[
            "rounds"
        ] = rounds

        return state

    return solve