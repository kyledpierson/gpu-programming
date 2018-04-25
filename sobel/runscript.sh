#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=cs6235
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp
#SBATCH -o sobel.out
#SBATCH -e sobel.err

ulimit -c unlimited -s

./sobel
# nvprof --analysis-metrics -o analysis.nvvp ./sobel
# nvprof --print-gpu-trace --metrics inst_integer,flop_count_dp,dram_read_throughput,dram_write_throughput -o analysis.nvvp ./sobel

