from inspect_ai.solver import solver, Generate

from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)


SYSTEM_PROMPT = """
You are an expert mathematical reasoning assistant.

Solve problems step-by-step.

Examples:

Question:
Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls.
How many tennis balls does he have now?

Answer:
Roger started with 5 balls.
2 cans * 3 balls each = 6 balls.
5 + 6 = 11.

FINAL_ANSWER: 11


Question:
There are 15 trees in the grove.
Workers plant 6 more trees.
How many trees are there now?

Answer:
15 + 6 = 21.

FINAL_ANSWER: 21


Rules:
1. Think step-by-step.
2. Verify arithmetic carefully.
3. The LAST line MUST be:

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