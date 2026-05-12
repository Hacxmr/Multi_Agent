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
# Optimized for 8k context DeepSeek/Qwen reasoning models
# ============================================================

REASONING_MAX_TOKENS = 1200
CRITIQUE_MAX_TOKENS  = 900
SYNTHESIS_MAX_TOKENS = 600

CANDIDATE_SNIPPET_CHARS = 1200


# ============================================================
# SYSTEM PROMPTS
# ============================================================

REASONING_SYSTEM = """
You are an expert mathematical reasoning assistant.

Solve the problem carefully step-by-step.

IMPORTANT RULES:
1. Use concise but correct reasoning.
2. Avoid unnecessary explanation.
3. Verify arithmetic carefully.
4. Do not stop reasoning early.
5. The FINAL line MUST be:

#### <number>

Example:
#### 42
"""


CRITIQUE_SYSTEM = """
You are a mathematical critic.

Review the candidate solutions carefully.

IMPORTANT RULES:
1. Identify arithmetic mistakes.
2. Correct incorrect reasoning.
3. Keep reasoning concise.
4. Verify calculations independently.
5. The FINAL line MUST be:

#### <number>
"""


SYNTHESIS_SYSTEM = """
You are a mathematical judge.

Choose the most defensible answer.

IMPORTANT RULES:
1. Verify arithmetic independently.
2. Do not blindly trust majority.
3. Use concise reasoning.
4. The FINAL line MUST be:

#### <number>
"""


# ============================================================
# UTILITIES
# ============================================================

def extract_answer(
    text: str,
) -> Optional[str]:

    if text is None:
        return None

    text = text.replace(",", "")

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",

        r"FINAL ANSWER:\s*([-+]?\d*\.?\d+)",

        r"answer is\s*([-+]?\d*\.?\d+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            value = float(
                match.group(1)
            )

            if value.is_integer():
                return str(int(value))

            return str(value)

    return None


def strip_think_blocks(
    text: str,
) -> str:

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
    text: str,
    max_chars: int = CANDIDATE_SNIPPET_CHARS,
) -> str:

    cleaned = strip_think_blocks(
        text
    )

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[-max_chars:]


def majority_vote(
    answers,
):

    valid_answers = [

        a for a in answers
        if a is not None
    ]

    if not valid_answers:
        return "UNKNOWN"

    return Counter(
        valid_answers
    ).most_common(1)[0][0]


async def run_agent(

    model,

    system_prompt: str,

    user_prompt: str,

    temperature: float,

    max_tokens: int,
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

            stop_seqs=[
                "</think>"
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
# MULTI-AGENT DEBATE SOLVER
# ============================================================

@solver
def debate_solver(

    agents: int = 3,

    rounds: int = 2,

    use_synthesis_judge: bool = True,

    base_temperature: float = 0.15,

    temperature_spread: float = 0.05,
):

    async def solve(
        state,
        generate: Generate,
    ):

        model = get_model()

        problem = state.input

        # ====================================================
        # ROUND 1
        # Independent reasoning
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

        for round_idx in range(rounds):

            new_outputs = []

            new_answers = []

            snippets = "\n\n".join(

                [

                    f"""
Agent {i+1}
Answer: {ans or "MISSING"}

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

If another agent is more correct,
revise your answer.

Otherwise defend your answer.

Provide concise reasoning.
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
        # SYNTHESIS JUDGE
        # ====================================================

        final_answer = None

        synthesis_text = None

        if use_synthesis_judge:

            answer_roster = "\n".join(

                [

                    f"""
Agent {i+1}: {ans or "MISSING"}
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

Select the most defensible answer.
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

                f"""
Agent {i+1}: {ans or "MISSING"}
"""

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

#### {final_answer}
"""

        # ====================================================
        # METADATA
        # ====================================================

        state.metadata[
            "round_outputs"
        ] = round_outputs

        state.metadata[
            "round_answers"
        ] = round_answers

        state.metadata[
            "final_answer"
        ] = final_answer

        return state

    return solve