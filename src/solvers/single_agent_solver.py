from inspect_ai.solver import (
    solver,
    Generate,
)

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)


SYSTEM_PROMPT = """
You are an expert mathematical reasoning assistant.

Instructions:
1. Solve step-by-step.
2. Verify every arithmetic operation carefully.
3. Re-check the final answer before responding.
4. Be concise but accurate.

The FINAL line MUST be:

#### <number>
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

                top_p=0.95,

                max_tokens=512,
            ),
        )

        state.output.completion = (
            response.completion
        )

        return state

    return solve