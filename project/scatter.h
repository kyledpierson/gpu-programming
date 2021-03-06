#ifndef SCATTER_H
#define SCATTER_H

#define HALO_SIZE    3
#define KERNEL_SIZE  (2 * HALO_SIZE + 1)

#define BLOCKDIM_X   32
#define BLOCKDIM_Y   32
#define RESULT_STEPS 1
#define HALO_STEPS   1

#include "JobScheduler.h"
void scatter(float *image, JobScheduler *scheduler, std::string outFile,
             int x_size, int y_size, int bytes,
             int ds_x_size_1, int ds_y_size_1, int ds_bytes_1,
             int ds_x_size_2, int ds_y_size_2, int ds_bytes_2,
             bool fourier, bool separable, float *gaussian, float *morlet_1, float *morlet_2);
void initConsts();

#endif
