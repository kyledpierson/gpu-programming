#ifndef SCATTER_H
#define SCATTER_H

#define HALO_SIZE    3
#define KERNEL_SIZE  (2 * HALO_SIZE + 1)

#define BLOCKDIM_X   16
#define BLOCKDIM_Y   8
#define RESULT_STEPS 8
#define HALO_STEPS   1

#include "JobScheduler.h"
void scatter(float *image, JobScheduler*, char* outFile,
             int x_size, int y_size, int bytes,
             int ds_x_size_1, int ds_y_size_1, int ds_bytes_1,
             int ds_x_size_2, int ds_y_size_2, int ds_bytes_2, bool separable);
void initConsts();

#endif
