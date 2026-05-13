

VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 2048 \
  --enforce-eager

inspect eval src/tasks/gsm8k_task.py@gsm8k_debate \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 15


export PYTHONPATH=$PYTHONPATH:$(pwd)

inspect eval src/tasks/gsm8k_task.py@gsm8k_single \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 20

  ssh -i $HOME\.ssh\id_ed25519 ubuntu@192.9.134.243

  python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192

  inspect view start --log-dir logs

  inspect eval src/tasks/gsm8k_task.py@gsm8k_majority_vote \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 15

inspect eval src/tasks/mmlu_task.py@mmlu_majority_vote \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 50


inspect eval src/tasks/truthfulqa_task.py@truthfulqa_majority_vote \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 50


# ============================================================
# START VLLM SERVER
# ============================================================

VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --enforce-eager


# ============================================================
# EXPORT PYTHONPATH
# ============================================================

export PYTHONPATH=$PYTHONPATH:$(pwd)

# ============================================================
# GSM8K — SINGLE AGENT
# ============================================================

inspect eval src/tasks/gsm8k_task.py@gsm8k_single \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 20


# ============================================================
# TRUTHFULQA — SINGLE AGENT
# ============================================================

inspect eval src/tasks/truthfulqa_task.py@truthfulqa_single \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 50


# ============================================================
# MMLU — SINGLE AGENT
# ============================================================

inspect eval src/tasks/mmlu_task.py@mmlu_single \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 10


  export PATH="$HOME/.local/bin:$PATH"


ssh -i $HOME/.ssh/id_ed25519 ubuntu@146.235.234.30

python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --enforce-eager

pkill -f vllm

abstract_algebra
formal_logic
college_mathematics
machine_learning
philosophy

abstract_algebra
formal_logic
college_mathematics
machine_learning
philosophy
computer_security
electrical_engineering
high_school_physics
college_physics
college_chemistry
econometrics
statistics
conceptual_physics
logical_fallacies
moral_disputes
professional_law
international_law
high_school_mathematics
high_school_statistics

## python -c "import src.tasks.mmlu_task"

inspect eval src/tasks/mmlu_task.py@mmlu_single \
  -T subject=abstract_algebra \
  --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --model-base-url http://localhost:8000/v1 \
  --limit 15