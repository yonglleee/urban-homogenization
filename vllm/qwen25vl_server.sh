#!/bin/bash


# 3. 启动 vllm 服务并置于后台
echo "Starting vllm service (port 8080) in the background..."
vllm serve "/data_nas/liyong/init_ckpt/Qwen2.5-VL-7B-Instruct" \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "Qwen2.5-VL-7B-Instruct" \
    --api-key EMPTY \
    --gpu-memory-utilization 0.90 \
    --max_model_len 10000 \
    --limit-mm-per-prompt '{"image": 1}' &
    # --limit-mm-per-prompt "image=1" &  这俩都尝试一下
    # vllm expects a JSON string like '{"image": 1}', not 'image=1'.

# 捕获服务的PID
VLLM_PID=$!
echo "VLLM service started with PID: $VLLM_PID"

# 3. 等待后台服务进程结束
echo "Script is now waiting for the service (PID: $VLLM_PID) to finish..." # <--- 修正点2: 修改注释为单数
wait $VLLM_PID 

echo "VLLM service has stopped. Job finished."