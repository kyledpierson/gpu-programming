#include <complex.h>
#include "cuda.h"
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <unistd.h>

#include "scatter.h"

// ============================= HELPER FUNCTIONS =============================
dim3 num_blocks(int x_size, int y_size, int x_threads, int y_threads) {
    // Compute the number of blocks needed for entire image
    if (x_size % x_threads) {
        x_size = x_size/x_threads*x_threads + x_threads;
    }
    if (y_size % y_threads) {
        y_size = y_size/y_threads*y_threads + y_threads;
    }
    int x_blocks = x_size / x_threads;
    int y_blocks = y_size / y_threads;

    dim3 blocks(x_blocks, y_blocks);
    return blocks;
}

__global__ void downsample(int *image, int *ds_image, int x_size, int ds_x_size) {
    int x_offset = blockIdx.x*BLOCKDIM_X+threadIdx.x;
    int y_offset = blockIdx.y*BLOCKDIM_Y+threadIdx.y;

    // Save every other pixel in downsampled image
    ds_image[y_offset*ds_x_size + x_offset] = image[2*(y_offset*x_size + x_offset)];
}

// ============================= KERNEL FUNCTIONS =============================
__device__ float convolution_pixel_2D(unsigned int tile[BLOCKDIM_Y][BLOCKDIM_X+1], float filter[KERNEL_SIZE][KERNEL_SIZE], int x, int y) {
    float value = 0;

    // Compute convolution
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            value += tile[y-HALO_SIZE+i][x-HALO_SIZE+j]*filter[KERNEL_SIZE-i-1][KERNEL_SIZE-j-1];
        }
    }

    return value;
}

__global__ void convolution_2D(unsigned int *image, int *result, int x_size, int y_size) {
    float gaussian_2D[7][7] = {
        {0.00000019425474,  0.000096568274, 0.00010062644,  0.00021978836,  0.00010062644,  0.000096568274, 0.00000019425474},
        {0.0000096568274,	0.00048006195,	0.0050023603,	0.010926159,	0.0050023603,	0.00048006195,	0.0000096568274},
        {0.00010062644,     0.0050023603,	0.052125789,    0.11385319,	    0.052125789,	0.0050023603,	0.00010062644},
        {0.00021978836,     0.010926159,	0.11385319,	    0.24867822,	    0.11385319,	    0.010926159,    0.00021978836},
        {0.00010062644,     0.0050023603,	0.052125789,    0.11385319,	    0.052125789,	0.0050023603,	0.00010062644},
        {0.0000096568274,	0.00048006195,	0.0050023603,	0.010926159,	0.0050023603,	0.00048006195,	0.0000096568274},
        {0.00000019425474,  0.000096568274, 0.00010062644,  0.00021978836,  0.00010062644,  0.000096568274, 0.00000019425474},
    };

    // Shared memory tile for image data
    __shared__ unsigned int tile[BLOCKDIM_Y][BLOCKDIM_X+1];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int offset = (blockIdx.y*(blockDim.y-(2*HALO_SIZE))+y)*x_size + (blockIdx.x*(blockDim.x-(2*HALO_SIZE))+x);

    // Load into shared memory
    tile[y][x] = image[offset];
    __syncthreads();

    // Each interior thread computes output
    if (x>=HALO_SIZE && x<blockDim.x-HALO_SIZE && y>=HALO_SIZE && y<blockDim.y-HALO_SIZE) {
        result[offset] = convolution_pixel_2D(tile, gaussian_2D, x, y);
    }
}

// ============================================================================
// ============================================================================
// ============================================================================
void scatter(unsigned int *image, int *result, int x_size, int y_size, int bytes, int ds_x_size, int ds_y_size, int ds_bytes) {
    // ====================== VARIABLES FOR CONVOLUTION =======================
    int x_active = BLOCKDIM_X-(2*HALO_SIZE);
    int y_active = BLOCKDIM_Y-(2*HALO_SIZE);

    dim3 blocks = num_blocks(x_size, y_size, x_active, y_active);
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

    // Allocate memory
    unsigned int *d_image;
    cudaMalloc((unsigned int**) &d_image, bytes);
    cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice);

    int *d_result;
    cudaMalloc((int**) &d_result, bytes);
    cudaMemset(d_result, 0, bytes);

    // ====================== VARIABLES FOR DOWNSAMPLING ======================
    dim3 ds_blocks = num_blocks(ds_x_size, ds_y_size, BLOCKDIM_X, BLOCKDIM_Y);

    // Allocate memory
    int *ds_result;
    cudaMalloc((int**) &ds_result, ds_bytes);
    cudaMemset(ds_result, 0, ds_bytes);

    // ===================== CONVOLUTION AND DOWNSAMPLING =====================
    float elapsed_time;
    cudaEvent_t start,stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Convolve and downsample
    convolution_2D<<<blocks, threads>>>(d_image, d_result, x_size, y_size);
    downsample<<<ds_blocks, threads>>>(d_result, ds_result, x_size, ds_x_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time,start, stop);

    // Copy the result
    cudaMemcpy(result, ds_result, ds_bytes, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_image);
    cudaFree(d_result);
    cudaFree(ds_result);

    fprintf(stderr, "TIME: %4.4f\n", elapsed_time);
}

