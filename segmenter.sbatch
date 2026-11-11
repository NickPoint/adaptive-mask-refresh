#!/bin/bash
#SBATCH --job-name=segmenter
#SBATCH --gres=gpu:tesla
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/segmenter-%j.out
#SBATCH --partition=gpu

set -euo pipefail

module load python
source .venv/bin/activate

export SEGMENTER_AUTHKEY=$(cat ~/.segmenter_key)
export SEGMENTER_HOST=0.0.0.0

PORT=${SEGMENTER_PORT:-6000}
SEGMENTERS=${SEGMENTERS:-1}

for ((k = 0; k < SEGMENTERS; k++)); do
    SEGMENTER_PORT=$((PORT + k)) python segmenter.py 2>&1 | sed -u "s/^/[$((PORT + k))] /" &
done
wait
