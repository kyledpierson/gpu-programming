#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --job-name=scatter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --qos=notchpeak-gpu
#SBATCH -o scatter.out
#SBATCH -e scatter.err

ulimit -c unlimited -s

./scatter
# nvprof --analysis-metrics -o analysis.nvvp ./scatter

