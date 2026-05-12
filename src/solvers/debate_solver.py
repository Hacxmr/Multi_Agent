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
# OPTIMAL SETTINGS FOR
# DeepSeek-R1-Distill-Qwen-7B + GSM8K
# ============================================================

BASE_REASONING_MAX_TOKENS = 700
HARD_REASONING_MAX_TOKENS = 1100

SYNTHESIS_MAX_TOKENS = 350

CANDIDATE_SNIPPET_CHARS = 700


# ============================================================
# SHORT HIGH-PERFORMANCE PROMPTS
# ============================================================

REASONING_PROMPTS = [

"""
Solve the math problem step-by-step briefly.

Keep reasoning concise.

The LAST line MUST be:

#### <number>

Example:
#### 42
""",

"""
Carefully solve the math problem.

Keep reasoning short and clear.

The LAST line MUST be:

#### <number>

Example:
#### 42
""",

"""
Solve the problem carefully using concise reasoning.

Do not add unnecessary explanation.

The LAST line MUST be:

#### <number>

Example:
#### 42
""",

"""
Compute the correct answer step-by-step.

Keep reasoning compact.

The LAST line MUST be:

#### <number>

Example:
#### 42
""",

"""
Solve carefully and avoid arithmetic mistakes.

Keep reasoning concise.

The LAST line MUST be:

#### <number>

Example:
#### 42
""",
]


# ============================================================
# TIE-BREAK JUDGE
# ============================================================

SYNTHESIS_SYSTEM = """
You are a mathematical judge.

You will receive multiple candidate solutions.

Select the MOST correct answer.

Keep reasoning concise.

The LAST line MUST be:

#### <number>

Example:
#### 42
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
# STRICT + STABLE
# ============================================================

def extract_answer(text: str) -> Optional[str]:

    if text is None:
        return None

    text = str(text)

    text = text.strip()

    # --------------------------------------------------------
    # STRICT #### EXTRACTION
    # --------------------------------------------------------

    match = re.findall(

        r"####\s*([-+]?\d*\.?\d+)",

        text,
    )

    if match:

        return normalize_number(
            match[-1]
        )

    # --------------------------------------------------------
    # BOXED ANSWER
    # --------------------------------------------------------

    match = re.findall(

        r"\\boxed\{([-+]?\d*\.?\d+)\}",

        text,
    )

    if match:

        return normalize_number(
            match[-1]
        )

    # --------------------------------------------------------
    # LAST NUMERIC LINE ONLY
    # --------------------------------------------------------

    lines = text.splitlines()

    for line in reversed(lines):

        line = line.strip()

        exact_match = re.fullmatch(

            r"[-+]?\d*\.?\d+",

            line,
        )

        if exact_match:

            return normalize_number(
                exact_match.group(0)
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
# DYNAMIC TOKEN BUDGET
# ============================================================

def get_reasoning_tokens(problem):

    word_count = len(problem.split())

    # --------------------------------------------------------
    # HARDER PROBLEMS
    # --------------------------------------------------------

    if word_count > 80:

        return HARD_REASONING_MAX_TOKENS

    return BASE_REASONING_MAX_TOKENS


# ============================================================
# FALLBACK EXTRACTION
# ============================================================

async def llm_extract_answer(

    model,

    text,
):

    prompt = f"""
Extract ONLY the final numeric answer.

Return EXACTLY:

#### <number>

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

            stop=["\n\n"],
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

            stop=[

                "\n\n\n",

                "Problem:",
            ],
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

    base_temperature=0.15,

    temperature_spread=0.10,
):

    """
    Optimized GSM8K Solver

    Designed specifically for:
    - DeepSeek-R1-Distill-Qwen-7B
    - self-consistency reasoning
    - stable extraction
    - concise CoT
    """

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        fallback_extractions = 0

        candidate_outputs = []

        candidate_answers = []

        # ====================================================
        # DYNAMIC TOKEN BUDGET
        # ====================================================

        reasoning_tokens = get_reasoning_tokens(
            problem
        )

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

                max_tokens=reasoning_tokens,
            )

            # ------------------------------------------------
            # FALLBACK EXTRACTION
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

Return EXACTLY:

#### <number>
"""

            synthesis_text, judge_answer = await run_agent(

                model=model,

                system_prompt=SYNTHESIS_SYSTEM,

                user_prompt=judge_prompt,

                temperature=0.0,

                max_tokens=SYNTHESIS_MAX_TOKENS,
            )

            # ------------------------------------------------
            # JUDGE FALLBACK EXTRACTION
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
            "reasoning_tokens"
        ] = reasoning_tokens

        state.metadata[
            "agents"
        ] = agents

        state.metadata[
            "rounds"
        ] = rounds

        return state

    return solve