#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --job-name=rg_cf_ssh
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ssh -o BatchMode=yes -o ConnectTimeout=30 CCDS-TC2 \
    'hostname; test -d /home/msai/lius0131/AGI/repostguard-lite'

