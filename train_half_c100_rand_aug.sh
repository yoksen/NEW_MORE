PYTHON="/root/miniconda3/envs/more/bin/python"
# for lr in 0.1 0.01 0.001; do
${PYTHON} train_half_c100_rand_aug.py --lr=0.1 --device=2 --epochs=600

# done