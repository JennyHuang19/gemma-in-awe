#!/bin/bash
# Experiment 2: self-chosen problem (play) -> solve under threat (work).
# Usage: sbatch run_self_chosen.sbatch <model_id> [extra args for run_self_chosen.py]
#   e.g. sbatch run_self_chosen.sbatch google/gemma-3-27b-it
#        sbatch run_self_chosen.sbatch google/gemma-4-31B-it --context continued
#SBATCH -J self-chosen
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
python experiments/self-chosen/run_self_chosen.py --model "$@"
