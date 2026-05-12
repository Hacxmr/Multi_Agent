from inspect_ai.solver import solver, Generate

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)


SYSTEM_PROMPT = """
You are an expert mathematical reasoning assistant.

You solve grade-school math problems carefully and accurately.

Rules:
1. Think step-by-step.
2. Show calculations clearly.
3. Verify arithmetic carefully.
4. The final line MUST be:
Final Answer: <number>
"""


@solver
def single_agent_solver():

    async def solve(state, generate: Generate):

        model = get_model()

        messages = [
            ChatMessageSystem(
                content=SYSTEM_PROMPT
            ),
            ChatMessageUser(
                content=state.input
            ),
        ]

        response = await model.generate(
            messages,
            config=GenerateConfig(
                temperature=0.2,
                max_tokens=512,
            ),
        )

        state.output.completion = (
            response.completion
        )

        return state

    return solve