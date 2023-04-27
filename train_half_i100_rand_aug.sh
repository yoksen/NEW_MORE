PYTHON="/root/miniconda3/envs/more/bin/python"
# for lr in 0.1 0.01 0.001; do
${PYTHON} train_half_i100_rand_aug.py --lr=0.001 --device=2 &

# done