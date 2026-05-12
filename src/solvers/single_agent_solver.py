from inspect_ai.solver import solver, Generate
from inspect_ai.model import get_model


@solver
def single_agent_solver():

    async def solve(state, generate: Generate):

        model = get_model()

        prompt = f"""
You are an expert mathematical reasoning assistant.

Solve the following grade-school math problem carefully.

Rules:
1. Think step-by-step.
2. Show intermediate calculations.
3. Double-check arithmetic.
4. The final line MUST be:
Final Answer: <number>

Question:
{state.input}
"""

        response = await model.generate(
            prompt,
            temperature=0.2,
        )

        state.output.completion = (
            response.completion
        )

        return state

    return solve