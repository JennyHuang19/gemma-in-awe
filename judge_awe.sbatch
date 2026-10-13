#!/bin/bash
# Experiment 1: LLM-judge awe/euphoria/joy over all play/work responses.
# Usage: sbatch judge_awe.sbatch [extra args for judge_awe.py]
#SBATCH -J judge-awe
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --requeue

export HF_HOME=$HOME/orcd/scratch/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_HUB_OFFLINE=1
export TQDM_DISABLE=1

source $HOME/envs/gemma/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python judge_awe.py "$@"
