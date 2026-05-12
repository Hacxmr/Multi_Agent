

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