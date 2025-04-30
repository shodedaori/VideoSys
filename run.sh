num_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_gpu=$(nvidia-smi --list-gpus | wc -l)
echo "Number of GPUs: $num_gpu"
torchrun --nnodes=1 --nproc-per-node=$num_gpu \
    vbench_delta.py
# python experiments/opensora.py