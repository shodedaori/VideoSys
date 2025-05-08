models=('opensora' 'cogvideox' 'latte')
methods=('base' 'pab' 'stu_1/2' 'stu_1/3' 'stu_1/5')

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        echo "Running model: ${model}, method: ${method}"
        python -m torch.distributed.run --nnodes=1 --nproc-per-node=1 \
            experiments/profile_model.py \
            --model ${model} \
            --method ${method}
    done
done
