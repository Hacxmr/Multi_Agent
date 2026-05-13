

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