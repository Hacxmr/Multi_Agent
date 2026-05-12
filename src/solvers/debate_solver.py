import re

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
1. Do not blindly follow other agents.
2. Verify calculations independently.
3. Revise ONLY if reasoning is stronger.
4. Keep reasoning concise.
5. The LAST line MUST be:

#### <number>
"""


JUDGE_PROMPT = """
You are the final mathematical judge.

Carefully compare all candidate answers.

IMPORTANT:
1. Do NOT blindly follow majority.
2. Independently verify arithmetic.
3. Choose the most correct solution.
4. Keep reasoning concise.
5. The LAST line MUST be:

#### <number>
"""


def extract_final_answer(text):

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

    return "UNKNOWN"


def compress_solution(solution):

    answer = extract_final_answer(
        solution
    )

    lines = solution.strip().split("\n")

    shortened = "\n".join(
        lines[-6:]
    )

    return f"""
{shortened}

Final Answer:
{answer}
"""


@solver
def debate_solver(
    rounds=3,
    agents=3,
):

    async def solve(state, generate: Generate):

        model = get_model()

        agent_solutions = []

        # ROUND 1
        # Independent reasoning

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

                    temperature=0.2 + (
                        agent_id * 0.1
                    ),

                    top_p=0.95,

                    max_tokens=512,
                ),
            )

            agent_solutions.append({

                "agent": f"Agent_{agent_id+1}",

                "solution": response.completion,
            })

        # ROUNDS 2-3

        for round_num in range(1, rounds):

            revised_solutions = []

            for agent_id in range(agents):

                peer_context = []

                for solution in agent_solutions:

                    if solution["agent"] != f"Agent_{agent_id+1}":

                        compressed = compress_solution(
                            solution["solution"]
                        )

                        peer_context.append(

                            f"""
{solution['agent']}:

{compressed}
"""
                        )

                debate_prompt = f"""
Question:
{state.input}

Other Agent Solutions:
{''.join(peer_context)}

Review the other solutions carefully.

If another solution is better,
revise your answer.

Otherwise defend your reasoning.

Provide concise updated reasoning.
"""

                messages = [

                    ChatMessageSystem(
                        content=SYSTEM_PROMPT
                    ),

                    ChatMessageUser(
                        content=debate_prompt
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

                revised_solutions.append({

                    "agent": f"Agent_{agent_id+1}",

                    "solution": response.completion,
                })

            agent_solutions = revised_solutions

        # FINAL JUDGE

        compressed_solutions = []

        for solution in agent_solutions:

            compressed = compress_solution(
                solution["solution"]
            )

            compressed_solutions.append(

                f"""
{solution['agent']}:

{compressed}
"""
            )

        judge_prompt = f"""
Question:
{state.input}

Candidate Solutions:
{''.join(compressed_solutions)}

Determine the most correct answer independently.
"""

        judge_messages = [

            ChatMessageSystem(
                content=JUDGE_PROMPT
            ),

            ChatMessageUser(
                content=judge_prompt
            ),
        ]

        final_response = await model.generate(

            judge_messages,

            config=GenerateConfig(

                temperature=0.1,

                top_p=0.9,

                max_tokens=512,
            ),
        )

        state.output.completion = (
            final_response.completion
        )

        return state

    return solve