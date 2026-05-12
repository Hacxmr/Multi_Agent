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

Solve problems carefully step-by-step.

IMPORTANT:
1. Do not stop reasoning early.
2. Verify calculations before finalizing.
3. The LAST line MUST be:

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

                max_tokens=1024,
            ),
        )

        state.output.completion = (
            response.completion
        )

        return state

    return solve