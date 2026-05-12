import re
import json

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


def extract_answer(text):

    match = re.search(
        r"####\s*([-+]?\d*\.?\d+)",
        text,
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


AGENT_SYSTEM_PROMPT = """
You are an expert mathematical reasoning agent.

IMPORTANT RULES:
1. Solve independently.
2. Do NOT blindly follow other agents.
3. Critically analyze previous reasoning.
4. Identify arithmetic mistakes explicitly.
5. Revise ONLY if evidence supports it.
6. The FINAL line MUST be:

#### <number>
"""


CRITIC_SYSTEM_PROMPT = """
You are a mathematical critic.

Your task:
1. Analyze the proposed answer.
2. Identify logical or arithmetic flaws.
3. Explain mistakes clearly.
4. Do NOT solve from scratch.
"""


JUDGE_SYSTEM_PROMPT = """
You are the final judge.

Rules:
1. Compare all candidate solutions carefully.
2. Prioritize arithmetic correctness.
3. Do NOT blindly follow the majority.
4. Independently verify calculations.
5. The FINAL line MUST be:

#### <number>
"""


@solver
def debate_solver(rounds=3, agents=3):

    async def solve(state, generate: Generate):

        model = get_model()

        agent_histories = []

        current_answers = []

        # ROUND 1 — independent reasoning

        for agent_id in range(agents):

            messages = [

                ChatMessageSystem(
                    content=AGENT_SYSTEM_PROMPT
                ),

                ChatMessageUser(
                    content=state.input
                ),
            ]

            response = await model.generate(

                messages,

                config=GenerateConfig(

                    temperature=0.3 + (
                        0.1 * agent_id
                    ),

                    top_p=0.95,

                    max_tokens=512,
                ),
            )

            answer = extract_answer(
                response.completion
            )

            current_answers.append({

                "agent": f"Agent_{agent_id+1}",

                "reasoning": response.completion,

                "answer": answer,
            })

        agent_histories.append(
            current_answers
        )

        # ROUNDS 2-3 critique + revision

        for round_num in range(1, rounds):

            revised_answers = []

            for agent_id in range(agents):

                peer_context = []

                for other in current_answers:

                    if other["agent"] != f"Agent_{agent_id+1}":

                        peer_context.append(

                            f"""
{other['agent']} proposed:

{other['reasoning']}
"""
                        )

                critique_prompt = f"""
Question:
{state.input}

Other Agent Solutions:
{''.join(peer_context)}

Review these carefully.

Identify flaws or confirm correctness.

Then provide your revised solution.
"""

                messages = [

                    ChatMessageSystem(
                        content=AGENT_SYSTEM_PROMPT
                    ),

                    ChatMessageUser(
                        content=critique_prompt
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

                answer = extract_answer(
                    response.completion
                )

                revised_answers.append({

                    "agent": f"Agent_{agent_id+1}",

                    "reasoning": response.completion,

                    "answer": answer,
                })

            current_answers = revised_answers

            agent_histories.append(
                revised_answers
            )

        debate_summary = json.dumps(
            current_answers,
            indent=2,
        )

        judge_prompt = f"""
Question:
{state.input}

Final Candidate Solutions:
{debate_summary}

Determine the correct answer independently.
"""

        judge_messages = [

            ChatMessageSystem(
                content=JUDGE_SYSTEM_PROMPT
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