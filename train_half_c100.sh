PYTHON="/root/miniconda3/envs/more/bin/python"
for lr in 0.1 0.01 0.001; do
    ${PYTHON} train_half_c100.py --lr=${lr}
done