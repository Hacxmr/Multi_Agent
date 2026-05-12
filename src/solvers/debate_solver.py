from inspect_ai.solver import solver, Generate
from inspect_ai.model import get_model

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
3. If previous agents made mistakes, explain why.
4. You may revise your answer.
5. Final line MUST be:
Final Answer: <number>
"""

                response = await model.generate(
                    prompt,
                    temperature=0.3,
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

Your task:
1. Analyze all proposed solutions.
2. Identify arithmetic mistakes.
3. Choose the most logically correct solution.
4. Final line MUST be:
Final Answer: <number>
"""

        final_response = await model.generate(
            judge_prompt,
            temperature=0.1,
        )

        state.output.completion = (
            final_response.completion
        )

        return state

    return solve