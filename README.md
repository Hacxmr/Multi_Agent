

VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 2048 \
  --enforce-eager

  