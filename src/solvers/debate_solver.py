from inspect_ai.solver import solver, Generate

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)

import json


AGENT_SYSTEM_PROMPT = """
You are an expert mathematical reasoning agent participating in a debate.

Rules:
1. Solve step-by-step.
2. Verify arithmetic carefully.
3. Critically evaluate previous solutions.
4. Revise your answer if needed.
5. Final line MUST be:
Final Answer: <number>
"""


JUDGE_SYSTEM_PROMPT = """
You are the final judge in a mathematical debate.

Rules:
1. Analyze all proposed solutions carefully.
2. Detect arithmetic mistakes.
3. Select the most logically correct solution.
4. Final line MUST be:
Final Answer: <number>
"""


@solver
def debate_solver(rounds=3, agents=3):

    async def solve(state, generate: Generate):

        model = get_model()

        debate_history = []

        context = ""

        for round_num in range(rounds):

            round_outputs = []

            for agent_id in range(agents):

                user_prompt = f"""
Question:
{state.input}

Previous Debate:
{context}
"""

                messages = [
                    ChatMessageSystem(
                        content=AGENT_SYSTEM_PROMPT
                    ),
                    ChatMessageUser(
                        content=user_prompt
                    ),
                ]

                response = await model.generate(
                    messages,
                    config=GenerateConfig(
                        temperature=0.3,
                        max_tokens=512,
                    ),
                )

                round_outputs.append(
                    {
                        "agent": f"Agent_{agent_id+1}",
                        "response": response.completion,
                    }
                )

            debate_history.append(
                round_outputs
            )

            context = json.dumps(
                round_outputs,
                indent=2,
            )

        judge_prompt = f"""
Question:
{state.input}

Debate History:
{json.dumps(debate_history, indent=2)}
"""

        judge_messages = [
            ChatMessageSystem(
                content=JUDGE_SYSTEM_PROMPT
            ),
            ChatMessageUser(
                content=judge_prompt
            ),
        ]

        final_response = await model.generate(
            judge_messages,
            config=GenerateConfig(
                temperature=0.1,
                max_tokens=512,
            ),
        )

        state.output.completion = (
            final_response.completion
        )

        return state

    return solve