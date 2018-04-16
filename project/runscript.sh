#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=scatter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp
#SBATCH -o scatter.out
#SBATCH -e scatter.err

ulimit -c unlimited -s

./scatter
# nvprof --analysis-metrics -o analysis.nvvp ./scatter

