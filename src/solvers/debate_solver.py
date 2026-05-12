from inspect_ai.solver import solver, Generate

from inspect_ai.model import (
    get_model,
    GenerateConfig,
)

import json


@solver
def debate_solver(rounds=3, agents=3):

    async def solve(state, generate: Generate):

        model = get_model()

        debate_history = []

        context = ""

        for round_num in range(rounds):

            round_responses = []

            for agent_id in range(agents):

                prompt = f"""
You are Agent {agent_id+1} participating in a mathematical reasoning debate.

Question:
{state.input}

Previous Debate:
{context}

Instructions:
1. Solve step-by-step.
2. Verify arithmetic carefully.
3. Identify mistakes from previous agents if any.
4. You may revise your answer.
5. The FINAL line MUST be:

Final Answer: <number>
"""

                response = await model.generate(
                    prompt,
                    config=GenerateConfig(
                        temperature=0.3,
                        max_tokens=512,
                    ),
                )

                round_responses.append(
                    {
                        "agent": f"Agent_{agent_id+1}",
                        "response": response.completion,
                    }
                )

            debate_history.append(
                round_responses
            )

            context = json.dumps(
                round_responses,
                indent=2,
            )

        judge_prompt = f"""
You are the final judge.

Question:
{state.input}

Debate History:
{json.dumps(debate_history, indent=2)}

Instructions:
1. Analyze all proposed solutions.
2. Identify arithmetic mistakes.
3. Choose the most logically correct answer.
4. The FINAL line MUST be:

Final Answer: <number>
"""

        final_response = await model.generate(
            judge_prompt,
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