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

# ---------------------------------------------------------------------------
# Token budget constants  (max-model-len = 2048 total: prompt + generation)
# ---------------------------------------------------------------------------
# Independent reasoning: system (~40) + problem (~150) + generation = 2048
#   → keep generation ≤ 500 so the model has room to show one clean derivation.
# Critique: system (~50) + problem (~150) + 3 × trimmed_candidate (~150 each)
#   = ~650 prompt tokens → generation budget ≤ 1398, but cap at 350 to stay safe.
# Synthesis: system (~50) + problem (~150) + answer list (~100) → cap at 300.

REASONING_MAX_TOKENS = 500
CRITIQUE_MAX_TOKENS  = 350
SYNTHESIS_MAX_TOKENS = 300

# How many characters to keep from each agent output when building critique context.
# ~300 chars ≈ 75 tokens per candidate; 3 candidates → ~225 tokens total.
CANDIDATE_SNIPPET_CHARS = 300


# ---------------------------------------------------------------------------
# Prompts — kept deliberately short to preserve generation budget
# ---------------------------------------------------------------------------

REASONING_SYSTEM = (
    "You are a concise mathematical solver. "
    "Solve step by step. "
    "Final line MUST be: #### <number>"
)

CRITIQUE_SYSTEM = (
    "You are a mathematical critic. "
    "You will see a problem and short excerpts from other agents. "
    "Identify any errors, then give your own corrected answer. "
    "Final line MUST be: #### <number>"
)

SYNTHESIS_SYSTEM = (
    "You are a mathematical judge. "
    "Given a problem and candidate answers, pick the most defensible one "
    "and confirm with a one-line derivation. "
    "Final line MUST be: #### <number>"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[str]:
    for pattern in (
        r"####\s*([-+]?\d*\.?\d+)",
        r"\\boxed\{([-+]?\d*\.?\d+)\}",
    ):
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks emitted by DeepSeek-R1 distills.

    These can consume hundreds of tokens; stripping them before they are
    passed back as context is the single most important budget saver here.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def trim_candidate(text: str, max_chars: int = CANDIDATE_SNIPPET_CHARS) -> str:
    """Strip think blocks then keep only the tail of the visible reasoning.

    The tail is most likely to contain the final steps and answer, which is
    what critics actually need.
    """
    cleaned = strip_think_blocks(text)
    return cleaned[-max_chars:].strip() if len(cleaned) > max_chars else cleaned


def majority_vote(answers: list[Optional[str]]) -> str:
    valid = [a for a in answers if a is not None]
    return Counter(valid).most_common(1)[0][0] if valid else "UNKNOWN"


async def run_agent(
    model,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, Optional[str]]:
    response = await model.generate(
        [
            ChatMessageSystem(content=system),
            ChatMessageUser(content=user),
        ],
        config=GenerateConfig(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
        ),
    )
    completion = response.completion
    return completion, extract_answer(completion)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@solver
def debate_solver(
    agents: int = 3,
    debate_rounds: int = 1,
    use_synthesis_judge: bool = True,
    base_temperature: float = 0.3,
    temperature_spread: float = 0.1,
):
    """
    Multi-agent debate solver tuned for a 2048-token total sequence budget.

    Token budget strategy
    ---------------------
    Round 1  — Independent reasoning.
        Each agent sees only the problem.
        Generation capped at REASONING_MAX_TOKENS (500).

    Round 2+ — Debate / critique.
        Each agent receives trimmed snippets (~300 chars each) of the previous
        round's visible completions (think blocks stripped), not the full text.
        Generation capped at CRITIQUE_MAX_TOKENS (350).

    Synthesis — Optional judge call.
        Receives only the problem + a compact answer roster.
        Generation capped at SYNTHESIS_MAX_TOKENS (300).

    Fallback  — Majority vote over post-debate answers if the judge fails
        to parse a valid answer.
    """

    async def solve(state, generate: Generate):
        model   = get_model()
        problem = state.input

        # -- Round 1: independent reasoning ----------------------------------
        r1_outputs: list[str]           = []
        r1_answers: list[Optional[str]] = []

        for agent_id in range(agents):
            temp = base_temperature + agent_id * temperature_spread
            out, ans = await run_agent(
                model=model,
                system=REASONING_SYSTEM,
                user=problem,
                temperature=temp,
                max_tokens=REASONING_MAX_TOKENS,
            )
            r1_outputs.append(out)
            r1_answers.append(ans)

        # -- Debate rounds ----------------------------------------------------
        prev_outputs = r1_outputs
        prev_answers = r1_answers

        for round_idx in range(debate_rounds):
            # Build a compact candidate block — trimmed snippets only.
            snippets = "\n\n".join(
                f"Agent {i + 1} (answer: {ans or 'MISSING'}):\n"
                f"{trim_candidate(out)}"
                for i, (out, ans) in enumerate(zip(prev_outputs, prev_answers))
            )

            d_outputs: list[str]           = []
            d_answers: list[Optional[str]] = []

            for agent_id in range(agents):
                temp = base_temperature + agent_id * temperature_spread
                user_msg = (
                    f"PROBLEM:\n{problem}\n\n"
                    f"OTHER AGENTS (round {round_idx + 1}, excerpts):\n{snippets}\n\n"
                    "Critique any errors above, then give your revised answer."
                )
                out, ans = await run_agent(
                    model=model,
                    system=CRITIQUE_SYSTEM,
                    user=user_msg,
                    temperature=temp,
                    max_tokens=CRITIQUE_MAX_TOKENS,
                )
                d_outputs.append(out)
                d_answers.append(ans)

            prev_outputs = d_outputs
            prev_answers = d_answers

        # -- Synthesis judge --------------------------------------------------
        final_outputs = prev_outputs
        final_answers = prev_answers
        synthesis_text: Optional[str] = None
        final_answer:   Optional[str] = None

        if use_synthesis_judge:
            # Absolute minimum context: problem + answer roster only.
            answer_roster = "\n".join(
                f"Agent {i + 1}: {ans or 'MISSING'}"
                for i, ans in enumerate(final_answers)
            )
            judge_user = (
                f"PROBLEM:\n{problem}\n\n"
                f"CANDIDATE ANSWERS:\n{answer_roster}\n\n"
                "Select the most defensible answer and confirm with a brief derivation."
            )
            synthesis_text, final_answer = await run_agent(
                model=model,
                system=SYNTHESIS_SYSTEM,
                user=judge_user,
                temperature=0.0,
                max_tokens=SYNTHESIS_MAX_TOKENS,
            )

        # -- Fallback: majority vote ------------------------------------------
        if final_answer is None:
            final_answer = majority_vote(final_answers)

        # -- Write state ------------------------------------------------------
        answer_summary = "\n".join(
            f"  Agent {i + 1}: {a or 'MISSING'}"
            for i, a in enumerate(final_answers)
        )

        state.output.completion = "\n".join(filter(None, [
            "Post-debate answers:",
            answer_summary,
            f"\nJudge: {synthesis_text}" if synthesis_text else "",
            f"\n#### {final_answer}",
        ]))

        state.metadata.update({
            "r1_outputs":     r1_outputs,
            "r1_answers":     r1_answers,
            "final_outputs":  final_outputs,
            "final_answers":  final_answers,
            "synthesis_text": synthesis_text,
            "final_answer":   final_answer,
        })

        return state

    return solve