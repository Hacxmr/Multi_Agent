from inspect_ai.solver import solver, Generate

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)


SYSTEM_PROMPT = """
You are an expert mathematical reasoning assistant.

Solve carefully step-by-step.

IMPORTANT:
The LAST line MUST be EXACTLY:

FINAL_ANSWER: <number>

Do NOT include units.
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
                max_tokens=256,
            ),
        )

        state.output.completion = (
            response.completion
        )

        return state

    return solve