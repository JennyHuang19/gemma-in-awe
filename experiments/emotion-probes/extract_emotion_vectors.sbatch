#!/bin/bash
# Emotion-vector extraction (forward passes only, no generation -> short job).
# Usage: sbatch extract_emotion_vectors.sbatch <model_id> [extra args]
#   e.g. sbatch extract_emotion_vectors.sbatch google/gemma-3-27b-it
#        sbatch extract_emotion_vectors.sbatch google/gemma-4-31B-it
#SBATCH -J emo-vectors
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --requeue

export HF_HOME=$HOME/orcd/scratch/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_HUB_OFFLINE=1
export TQDM_DISABLE=1

source $HOME/envs/gemma/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python experiments/emotion-probes/extract_emotion_vectors.py --model "$@"
