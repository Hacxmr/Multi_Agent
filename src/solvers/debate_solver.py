import re

from inspect_ai.solver import solver, Generate

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)


def extract_answer(text):

    match = re.search(
        r"FINAL_ANSWER:\s*(-?\d+\.?\d*)",
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


AGENT_PROMPT = """
You are an expert mathematical reasoning agent.

Solve carefully step-by-step.

IMPORTANT:
The LAST line MUST be EXACTLY:

FINAL_ANSWER: <number>

Do NOT include units.
"""


JUDGE_PROMPT = """
You are the final mathematical judge.

You must independently determine the correct answer.

Do NOT blindly follow the majority.
"""


@solver
def debate_solver(rounds=3, agents=3):

    async def solve(state, generate: Generate):

        model = get_model()

        debate_summary = []

        for round_num in range(rounds):

            round_answers = []

            for agent_id in range(agents):

                context = "\n".join(
                    [
                        f"{x['agent']} answered {x['answer']}"
                        for x in debate_summary
                    ]
                )

                user_prompt = f"""
Question:
{state.input}

Previous Answers:
{context}
"""

                messages = [
                    ChatMessageSystem(
                        content=AGENT_PROMPT
                    ),
                    ChatMessageUser(
                        content=user_prompt
                    ),
                ]

                response = await model.generate(
                    messages,
                    config=GenerateConfig(
                        temperature=0.3,
                        max_tokens=256,
                    ),
                )

                answer = extract_answer(
                    response.completion
                )

                round_answers.append(
                    {
                        "agent": f"Agent_{agent_id+1}",
                        "answer": answer,
                    }
                )

            debate_summary.extend(
                round_answers
            )

        summary_text = "\n".join(
            [
                f"{x['agent']} proposed {x['answer']}"
                for x in debate_summary
            ]
        )

        judge_messages = [
            ChatMessageSystem(
                content=JUDGE_PROMPT
            ),
            ChatMessageUser(
                content=f"""
Question:
{state.input}

Debate Summary:
{summary_text}

Solve independently and produce the final answer.
"""
            ),
        ]

        final_response = await model.generate(
            judge_messages,
            config=GenerateConfig(
                temperature=0.1,
                max_tokens=256,
            ),
        )

        state.output.completion = (
            final_response.completion
        )

        return state

    return solve