import re
from collections import Counter

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

Solve carefully step-by-step.

IMPORTANT:
The FINAL line MUST be:

#### <number>
"""


def extract_answer(text):

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
        )

        if match:
            return match.group(1)

    return None


@solver
def debate_solver(
    agents=3,
):

    async def solve(state, generate: Generate):

        model = get_model()

        candidate_outputs = []

        candidate_answers = []

        # INDEPENDENT REASONING

        for agent_id in range(agents):

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

                    temperature=0.3 + (
                        agent_id * 0.1
                    ),

                    top_p=0.95,

                    max_tokens=1024,
                ),
            )

            completion = response.completion

            answer = extract_answer(
                completion
            )

            candidate_outputs.append(
                completion
            )

            candidate_answers.append(
                answer
            )

        # MAJORITY VOTE

        valid_answers = [

            a for a in candidate_answers
            if a is not None
        ]

        if valid_answers:

            majority_answer = Counter(
                valid_answers
            ).most_common(1)[0][0]

        else:

            majority_answer = "UNKNOWN"

        # FINAL OUTPUT

        final_output = f"""
Independent Candidate Answers:

{candidate_answers}

Final Selected Answer:

#### {majority_answer}
"""

        state.output.completion = (
            final_output
        )

        state.metadata[
            "candidate_outputs"
        ] = candidate_outputs

        return state

    return solve