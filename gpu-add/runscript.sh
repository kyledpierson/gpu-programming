#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=cs6235
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp
ulimit -c unlimited -s
./add
