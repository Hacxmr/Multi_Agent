from inspect_ai.solver import solver, Generate
from inspect_ai.model import get_model


@solver
def single_agent_solver():

    async def solve(state, generate: Generate):

        model = get_model()

        prompt = f"""
Question:
{state.input}

Solve carefully.

Return:
Final Answer: ...
"""

        response = await model.generate(prompt)

        state.output.completion = response.completion

        return state

    return solve