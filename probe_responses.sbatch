#!/bin/bash
# Experiment 2: emotion probes at first/last output token over a model's responses.
# Usage: sbatch probe_responses.sbatch <model_id> [extra args]
#   e.g. sbatch probe_responses.sbatch google/gemma-3-27b-it
#SBATCH -J probe-resp
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --requeue

export HF_HOME=$HOME/orcd/scratch/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_HUB_OFFLINE=1
export TQDM_DISABLE=1

source $HOME/envs/gemma/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python probe_responses.py --model "$@"
