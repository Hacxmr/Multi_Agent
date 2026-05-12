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
# FAST + STABLE SETTINGS
# Optimized for GSM8K + DeepSeek/Qwen
# Majority Vote + Optional Judge Tie-Break
# ============================================================

REASONING_MAX_TOKENS = 650
SYNTHESIS_MAX_TOKENS = 400

CANDIDATE_SNIPPET_CHARS = 700


# ============================================================
# SYSTEM PROMPTS
# ============================================================

REASONING_SYSTEM = """
You are an expert mathematical reasoning assistant.

Solve carefully step-by-step.

IMPORTANT:
1. Keep reasoning concise.
2. Verify arithmetic carefully.
3. The FINAL line MUST contain ONLY the numeric answer.

Example:
42
"""


SYNTHESIS_SYSTEM = """
You are a mathematical judge.

Choose the most correct answer.

IMPORTANT:
1. Verify calculations independently.
2. Do not blindly trust majority.
3. Keep reasoning concise.
4. The FINAL line MUST contain ONLY the numeric answer.

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
    # LAST NUMERIC LINE PREFERENCE
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
    # FINAL FALLBACK:
    # LAST NUMBER ANYWHERE
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


# ============================================================
# LLM EXTRACTION FALLBACK
# ============================================================

async def llm_extract_answer(

    model,

    text,
):

    prompt = f"""
Extract the final numeric answer from the following solution.

Return ONLY the final numeric value.

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

    # --------------------------------------------------------
    # LLM FALLBACK EXTRACTION
    # --------------------------------------------------------

    if answer is None:

        answer = await llm_extract_answer(

            model,

            completion,
        )

    return completion, answer


# ============================================================
# MAJORITY VOTE SOLVER
# ============================================================

@solver
def debate_solver(

    agents=5,

    use_synthesis_judge=True,

    base_temperature=0.15,

    temperature_spread=0.08,
):

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        # ====================================================
        # INDEPENDENT REASONING
        # ====================================================

        candidate_outputs = []

        candidate_answers = []

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
                    most_common[0][1] ==
                    most_common[1][1]
                ):

                    tie = True

        final_answer = voted_answer

        synthesis_text = ""

        # ====================================================
        # OPTIONAL JUDGE
        # Only if tie or unknown
        # ====================================================

        if use_synthesis_judge and (
            tie or final_answer == "UNKNOWN"
        ):

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

Determine the most correct answer.
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

                f"""
Agent {i+1}: {ans}
"""

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

Judge Decision:

{synthesis_text}

Final Answer:
{final_answer}
"""

        # ====================================================
        # METADATA
        # ====================================================

        state.metadata[
            "candidate_answers"
        ] = candidate_answers

        state.metadata[
            "final_answer"
        ] = final_answer

        state.metadata[
            "majority_vote"
        ] = voted_answer

        return state

    return solve