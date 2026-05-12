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

# Sweet spot for DeepSeek-R1-Distill-Qwen-7B
# Prevents truncation while avoiding excessive drift
REASONING_MAX_TOKENS = 700

# Keep metadata logs manageable
MAX_LOG_CHARS = 1500


# ============================================================
# PROMPTS
# ============================================================

# Strongly enforce structured final answers
REASONING_PROMPTS = [

"""
Solve the math problem carefully step-by-step.

You MUST end your response with:

#### final_numeric_answer
""",

"""
Reason carefully and avoid arithmetic mistakes.

The FINAL line MUST be:

#### answer
""",

"""
Solve step-by-step.

Finish EXACTLY with:

#### final_answer
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
# STRUCTURAL EXTRACTION
# ============================================================

def extract_answer(text: str) -> Optional[str]:

    if text is None:
        return None

    text = str(text)

    text = text.replace(",", "")
    text = text.replace("$", "")

    lines = text.splitlines()

    # ========================================================
    # LEVEL 1 — STRICT STRUCTURED EXTRACTION
    # ========================================================

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
        )

        if matches:

            return normalize_number(
                matches[-1]
            )

    # ========================================================
    # LEVEL 2 — LAST STANDALONE NUMERIC LINE
    # ========================================================

    for line in reversed(lines):

        line = line.strip()

        if re.fullmatch(

            r"[-+]?\d*\.?\d+",

            line,
        ):

            return normalize_number(line)

    # ========================================================
    # LEVEL 3 — SAFE LOCALIZED FALLBACK
    # ========================================================

    tail_lines = lines[-5:]

    candidate_numbers = []

    for line in tail_lines:

        nums = re.findall(

            r"[-+]?\d*\.?\d+",

            line,
        )

        for n in nums:

            val = normalize_number(n)

            if val is not None:

                try:

                    candidate_numbers.append(
                        float(val)
                    )

                except ValueError:

                    pass

    if candidate_numbers:

        # Prefer largest-magnitude tail value
        best_candidate = max(

            candidate_numbers,

            key=lambda x: abs(x)
        )

        return normalize_number(
            best_candidate
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
# LOG TRIMMING
# ============================================================

def trim_log(text, max_chars=MAX_LOG_CHARS):

    if text is None:
        return ""

    if len(text) <= max_chars:

        return text

    return text[:max_chars]


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

    rounds=1,  # backward compatibility

    base_temperature=0.15,

    temperature_spread=0.08,
):

    """
    Final optimized GSM8K self-consistency solver.

    Architecture:
    - 5 independent agents
    - majority voting
    - structural extraction
    - metadata reasoning logs
    - no judge
    - no verifier
    """

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        candidate_answers = []

        agent_logs = []

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

            candidate_answers.append(
                answer
            )

            # ------------------------------------------------
            # STORE LOGS IN METADATA ONLY
            # ------------------------------------------------

            agent_logs.append({

                "agent_id": agent_id + 1,

                "temperature": temperature,

                "answer": answer,

                "reasoning": trim_log(completion),
            })

        # ====================================================
        # MAJORITY VOTING
        # ====================================================

        valid_answers = [

            a for a in candidate_answers
            if a is not None
        ]

        answer_counts = Counter(
            valid_answers
        )

        voted_answer = "UNKNOWN"

        if len(answer_counts) > 0:

            voted_answer = answer_counts.most_common(1)[0][0]

        final_answer = voted_answer

        # ====================================================
        # LIGHTWEIGHT OUTPUT
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
        state.metadata["agent_logs"] = agent_logs
        state.metadata["agents"] = agents
        state.metadata["rounds"] = rounds

        return state

    return solve