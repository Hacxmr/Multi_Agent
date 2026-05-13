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


# ============================================================
# SETTINGS
# ============================================================

REASONING_MAX_TOKENS = 2000

MAX_LOG_CHARS = 1500


# ============================================================
# PROMPTS
# ============================================================

GSM8K_PROMPTS = [

"""
You are an expert mathematical reasoning assistant.

Solve the problem carefully step-by-step.

IMPORTANT:
The LAST line MUST be:

#### <number>
""",

"""
Reason carefully and avoid arithmetic mistakes.

IMPORTANT:
The LAST line MUST be:

#### <number>
""",
]


MMLU_PROMPTS = [

"""
You are an expert academic reasoning assistant.

Reason through each statement in the question carefully, then commit to a final answer.

Strict reasoning rules:
1. For each sub-statement, decide TRUE or FALSE before moving to the next.
2. If you find a counterexample proving a statement is FALSE, note it and stop exploring.
3. Keep reasoning clear and direct.

IMPORTANT:
The LAST line MUST be EXACTLY:

FINAL_ANSWER: A

(Replace A with B/C/D when appropriate.)
""",

"""
You are an expert reasoning assistant.

Carefully analyze the options. Work through your reasoning systematically.

For logical or mathematical questions: check your work step-by-step.
For arguments: identify premises and conclusions.
For definitions: match terms precisely to options.

IMPORTANT:
The LAST line MUST be EXACTLY:

FINAL_ANSWER: A

(Replace A with B/C/D when appropriate.)
""",
]


TRUTHFULQA_PROMPTS = [

"""
You are a truthful and factual assistant.

Avoid misinformation and unsupported claims.
""",

"""
Answer factually and concisely.

If uncertain, state uncertainty honestly.
""",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_number(value):

    try:

        value = str(value)

        value = value.replace(",", "")
        value = value.replace("$", "")
        value = value.strip()

        value = float(value)

        if value.is_integer():

            return str(int(value))

        return str(round(value, 6))

    except:

        return None


def normalize_text(text):

    if text is None:

        return None

    return str(text).strip().lower()


# ============================================================
# TASK DETECTION
# ============================================================

def detect_task(problem):

    text = str(problem).lower()

    # --------------------------------------------------------
    # MMLU
    # --------------------------------------------------------

    if "##mmlu##" in text:

        return "mmlu"

    # --------------------------------------------------------
    # TRUTHFULQA
    # --------------------------------------------------------

    if any(
        x in text
        for x in [
            "truthful",
            "misinformation",
            "myth",
            "conspiracy",
        ]
    ):

        return "truthfulqa"

    # --------------------------------------------------------
    # GSM8K DEFAULT
    # --------------------------------------------------------

    return "gsm8k"


# ============================================================
# GSM8K EXTRACTION
# ============================================================

def extract_numeric_answer(text):

    if text is None:

        return None

    text = str(text)

    patterns = [

        r"####\s*([-+]?\d*\.?\d+)",

        r"\\boxed\{([-+]?\d*\.?\d+)\}",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
        )

        if matches:

            return normalize_number(
                matches[-1]
            )

    numbers = re.findall(

        r"[-+]?\d*\.?\d+",

        text,
    )

    if numbers:

        return normalize_number(
            numbers[-1]
        )

    return None


# ============================================================
# MMLU EXTRACTION
# ============================================================

def extract_mcq_answer(text):

    if text is None:

        return None

    text = str(text)

    # --------------------------------------------------------
    # PRIORITY 1 — explicit answer markers
    # --------------------------------------------------------

    priority_patterns = [

        r"FINAL_ANSWER:\s*([A-D])",

        r"ANSWER:\s*([A-D])",

        r"answer\s+is\s*[:\s]*([A-D])\b",

        r"correct\s+(?:option|answer)\s+is\s*[:\s]*([A-D])\b",
    ]

    for pattern in priority_patterns:

        matches = re.findall(

            pattern,

            text,

            re.IGNORECASE,
        )

        if matches:

            return matches[-1].upper()

    # --------------------------------------------------------
    # PRIORITY 2 — last line exactly A/B/C/D
    # --------------------------------------------------------

    for line in reversed(text.splitlines()):

        line = line.strip()

        m = re.fullmatch(

            r"([A-D])[.\s]?",

            line,

            re.IGNORECASE,
        )

        if m:

            return m.group(1).upper()

    # --------------------------------------------------------
    # PRIORITY 3 — final standalone A/B/C/D
    # --------------------------------------------------------

    option_matches = re.findall(

        r"\b([A-D])\b",

        text,
    )

    if option_matches:

        return option_matches[-1].upper()

    return None


# ============================================================
# TRUTHFULQA EXTRACTION
# ============================================================

def extract_truthfulqa_answer(text):

    if text is None:

        return None

    return normalize_text(text)


# ============================================================
# THINK BLOCK REMOVAL
# ============================================================

def strip_think_blocks(text):

    if text is None:

        return ""

    text = re.sub(

        r"<think>.*?</think>",

        "",

        text,

        flags=re.DOTALL,
    )

    return text.strip()


# ============================================================
# LOG TRIMMING
# ============================================================

def trim_log(text, max_chars=MAX_LOG_CHARS):

    if text is None:

        return ""

    if len(text) <= max_chars:

        return text

    return text[:max_chars]


# ============================================================
# TASK-SPECIFIC EXTRACTION
# ============================================================

def extract_answer(text, task_type):

    if task_type == "gsm8k":

        return extract_numeric_answer(text)

    elif task_type == "mmlu":

        return extract_mcq_answer(text)

    elif task_type == "truthfulqa":

        return extract_truthfulqa_answer(text)

    return None


# ============================================================
# TASK-SPECIFIC PROMPTS
# ============================================================

def get_prompts(task_type):

    if task_type == "gsm8k":

        return GSM8K_PROMPTS

    elif task_type == "mmlu":

        return MMLU_PROMPTS

    elif task_type == "truthfulqa":

        return TRUTHFULQA_PROMPTS

    return GSM8K_PROMPTS


# ============================================================
# MODEL CALL
# ============================================================

async def run_agent(

    model,

    system_prompt,

    user_prompt,

    temperature,

    max_tokens,

    task_type,
):

    response = await model.generate(

        [

            ChatMessageSystem(
                content=system_prompt
            ),

            ChatMessageUser(
                content=user_prompt
            ),
        ],

        config=GenerateConfig(

            temperature=temperature,

            top_p=0.95,

            max_tokens=max_tokens,
        ),
    )

    completion = strip_think_blocks(
        response.completion
    )

    answer = extract_answer(

        completion,

        task_type,
    )

    return completion, answer


# ============================================================
# MAJORITY VOTE SOLVER
# ============================================================

@solver
def majority_vote_solver(

    agents=5,

    base_temperature=0.1,

    temperature_spread=0.05,
):

    async def solve(state, generate: Generate):

        model = get_model()

        problem = state.input

        task_type = detect_task(problem)

        prompts = get_prompts(
            task_type
        )

        candidate_answers = []

        agent_logs = []

        # ====================================================
        # AGENT GENERATION
        # ====================================================

        for agent_id in range(agents):

            temperature = (

                base_temperature +

                (
                    agent_id *
                    temperature_spread
                )
            )

            system_prompt = prompts[
                agent_id % len(prompts)
            ]

            completion, answer = await run_agent(

                model=model,

                system_prompt=system_prompt,

                user_prompt=problem,

                temperature=temperature,

                max_tokens=REASONING_MAX_TOKENS,

                task_type=task_type,
            )

            candidate_answers.append(
                answer
            )

            agent_logs.append({

                "agent_id": agent_id + 1,

                "temperature": temperature,

                "answer": answer,

                "reasoning": trim_log(
                    completion
                ),
            })

        # ====================================================
        # MAJORITY VOTE
        # ====================================================

        valid_answers = [

            a for a in candidate_answers
            if a is not None
        ]

        voted_answer = "UNKNOWN"

        if valid_answers:

            counts = Counter(
                valid_answers
            )

            voted_answer = counts.most_common(1)[0][0]

        final_answer = voted_answer

        # ====================================================
        # OUTPUT SUMMARY
        # ====================================================

        answer_summary = "\n".join(

            [

                f"Agent {i+1}: {ans}"

                for i, ans in enumerate(
                    candidate_answers
                )
            ]
        )

        # ====================================================
        # FINAL OUTPUT FORMAT
        # ====================================================

        if task_type == "gsm8k":

            final_output = f"""
Independent Agent Answers:

{answer_summary}

Majority Vote:
{final_answer}

#### {final_answer}
"""

        elif task_type == "mmlu":

            final_output = f"""
Independent Agent Answers:

{answer_summary}

Majority Vote:
{final_answer}

FINAL_ANSWER: {final_answer}
"""

        else:

            final_output = f"""
Independent Agent Answers:

{answer_summary}

Majority Vote:
{final_answer}
"""

        state.output.completion = final_output

        # ====================================================
        # METADATA
        # ====================================================

        state.metadata["task_type"] = task_type
        state.metadata["candidate_answers"] = candidate_answers
        state.metadata["majority_vote"] = voted_answer
        state.metadata["final_answer"] = final_answer
        state.metadata["agent_logs"] = agent_logs
        state.metadata["agents"] = agents

        return state

    return solve