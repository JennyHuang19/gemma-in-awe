#!/bin/bash
# Consolidate both overnight experiments into results/OVERNIGHT_SUMMARY.md.
# CPU-only; meant to run after the judge + probe jobs via --dependency=afterany.
# Usage: sbatch --dependency=afterany:<j1>:<j2>:<j3> summarize_overnight.sbatch
#SBATCH -J overnight-sum
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out

source $HOME/envs/gemma/bin/activate
python summarize_overnight.py
