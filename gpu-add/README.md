# CUDA Vector Add

## Files

* driver.cu : main source file
* Makefile
* runscript : batch script for chpc
* README.md : this document

## Compile and Run

CHPC uses module system to manage software and versions through `module` command. Here we need the following modules:

~~~
module load cuda/9.1  # For cuda toolchain, here we are using nvcc -
                      #   The NVIDIA CUDA Compiler
module load gcc/6.4.0 # For a newer version of gcc
~~~

You can use `module spider cuda` to check the available CUDA versions on CHPC. After loading necessary modules, we can build and run the program with:

~~~
make                  # Build executable in the source directory
sbatch runscript.sh   # Submit batch job on chpc
                      #   Return <job_id> on success
~~~

The output from the run will be saved in file `slurm-*.out`

You can use `squeue` to check the status of the job by either look for the job_id or your uid(login name).

