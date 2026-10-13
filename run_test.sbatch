#!/bin/bash
#SBATCH -J gemma-test
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out

export HF_HOME=$HOME/orcd/scratch/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_HUB_OFFLINE=1  # weights are pre-downloaded on the login node

source $HOME/envs/gemma/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python test_inference.py
