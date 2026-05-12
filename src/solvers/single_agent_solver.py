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

Solve step-by-step carefully.

VERY IMPORTANT:
The FINAL line MUST be EXACTLY:

#### <number>

Examples:
#### 42
#### 17
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