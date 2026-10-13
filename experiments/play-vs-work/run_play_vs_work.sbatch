#!/bin/bash
# Usage: sbatch run_play_vs_work.sbatch <model_id> [extra args for run_play_vs_work.py]
#   e.g. sbatch run_play_vs_work.sbatch google/gemma-3-27b-it
#        sbatch run_play_vs_work.sbatch google/gemma-4-31B-it --samples 3
#SBATCH -J play-vs-work
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --requeue

export HF_HOME=$HOME/orcd/scratch/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_HUB_OFFLINE=1
export TQDM_DISABLE=1

source $HOME/envs/gemma/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python experiments/play-vs-work/run_play_vs_work.py --model "$@"
