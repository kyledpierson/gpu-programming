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

./reduction -i Dataset/0/input.raw -o Dataset/0/output.raw
./reduction -i Dataset/1/input.raw -o Dataset/1/output.raw
./reduction -i Dataset/2/input.raw -o Dataset/2/output.raw
./reduction -i Dataset/3/input.raw -o Dataset/3/output.raw
./reduction -i Dataset/4/input.raw -o Dataset/4/output.raw
./reduction -i Dataset/5/input.raw -o Dataset/5/output.raw
./reduction -i Dataset/6/input.raw -o Dataset/6/output.raw
./reduction -i Dataset/7/input.raw -o Dataset/7/output.raw
./reduction -i Dataset/8/input.raw -o Dataset/8/output.raw
./reduction -i Dataset/9/input.raw -o Dataset/9/output.raw

