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
Question:
{state.input}

Previous Debate Context:
{context}

You are Agent {agent_id + 1}.

Carefully analyze the problem.

You may revise your answer based on previous discussion.

Return ONLY:

Reasoning: ...
Final Answer: ...
"""

                response = await model.generate(prompt)

                round_responses.append(
                    {
                        "agent": f"Agent_{agent_id+1}",
                        "response": response.completion,
                    }
                )

            debate_history.append(round_responses)

            context = json.dumps(
                round_responses,
                indent=2,
            )

        judge_prompt = f"""
Question:
{state.input}

Debate History:
{json.dumps(debate_history, indent=2)}

You are the final judge.

Choose the BEST final answer.

Return ONLY:

Final Answer: ...
"""

        final_response = await model.generate(
            judge_prompt
        )

        state.output.completion = (
            final_response.completion
        )

        return state

    return solve