PYTHON="/root/miniconda3/envs/more/bin/python"
for lr in 0.001 0.01 0.1; do
    ${PYTHON} train_half_i100.py --lr=${lr}
done