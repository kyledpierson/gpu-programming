#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuComplex.h>

#include "scatter.h"

// ============================= HELPER FUNCTIONS =============================
dim3 num_blocks_separable(int x_size, int y_size, int x_threads, int y_threads) {
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

__constant__ float d_kernel[KERNEL_SIZE];
void copy_kernel_1D(float h_kernel[KERNEL_SIZE]) {
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float));
}

// ============================= KERNEL FUNCTIONS =============================
__global__ void convolution_row(float *image, float *result, int x_size, int y_size) {
    __shared__ float tile[BLOCKDIM_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_X];

    // Offset to the left halo edge
    const int x_start = (blockIdx.x * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_X + threadIdx.x;
    const int y_start = blockIdx.y * BLOCKDIM_Y + threadIdx.y;

    image += y_start * x_size + x_start;
    result += y_start * x_size + x_start;

#pragma unroll
    // Load left halo
    for (int i = 0; i < HALO_STEPS; i++) {
        tile[threadIdx.y][threadIdx.x + i * BLOCKDIM_X] = (x_start >= -i * BLOCKDIM_X) ? image[i * BLOCKDIM_X] : 0;
    }

#pragma unroll
    // Load main data
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
        tile[threadIdx.y][threadIdx.x + i * BLOCKDIM_X] = image[i * BLOCKDIM_X];
    }

#pragma unroll
    // Load right halo
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++) {
        tile[threadIdx.y][threadIdx.x + i * BLOCKDIM_X] = (x_size - x_start > i * BLOCKDIM_X) ? image[i * BLOCKDIM_X] : 0;
    }

    __syncthreads();

#pragma unroll
    // Compute results
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
        float sum = 0;

#pragma unroll
        for (int j = -HALO_SIZE; j <= HALO_SIZE; j++) {
            sum += d_kernel[HALO_SIZE - j] * tile[threadIdx.y][threadIdx.x + i * BLOCKDIM_X + j];
        }
        result[i * BLOCKDIM_X] = sum;
    }
}

__global__ void convolution_col(float *image, float *result, int x_size, int y_size) {
    __shared__ float tile[BLOCKDIM_X][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Y + 1];

    // Offset to the upper halo edge
    const int x_start = blockIdx.x * BLOCKDIM_X + threadIdx.x;
    const int y_start = (blockIdx.y * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Y + threadIdx.y;
    image += y_start * x_size + x_start;
    result += y_start * x_size + x_start;

#pragma unroll
    //Upper halo
    for (int i = 0; i < HALO_STEPS; i++) {
        tile[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = (y_start >= -i * BLOCKDIM_Y) ? image[i * BLOCKDIM_Y * x_size] : 0;
    }

#pragma unroll
    //Main data
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
        tile[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = image[i * BLOCKDIM_Y * x_size];
    }

#pragma unroll
    //Lower halo
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++) {
        tile[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = (y_size - y_start > i * BLOCKDIM_Y) ? image[i * BLOCKDIM_Y * x_size] : 0;
    }

    __syncthreads();

#pragma unroll
    //Compute results
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
        float sum = 0;

#pragma unroll
        for (int j = -HALO_SIZE; j <= HALO_SIZE; j++) {
            sum += d_kernel[HALO_SIZE - j] * tile[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y + j];
        }
        result[i * BLOCKDIM_Y * x_size] = sum;
    }
}

__global__ void downsample_separable(float *image, float *ds_image, int x_size, int ds_x_size) {
    int x_offset = blockIdx.x*BLOCKDIM_X+threadIdx.x;
    int y_offset = blockIdx.y*BLOCKDIM_Y+threadIdx.y;

    // Save every other pixel in downsampled image
    ds_image[y_offset*ds_x_size + x_offset] = image[2*(y_offset*x_size + x_offset)];
}

void scatter_separable(float *image, float *result, int x_size, int y_size, int bytes, int ds_x_size, int ds_y_size, int ds_bytes) {
    float gaussian_1D[7] = {0.000395, 0.021639, 0.229031, 0.497871, 0.229031, 0.021639, 0.000395};
    copy_kernel_1D(gaussian_1D);

    // ====================== VARIABLES FOR CONVOLUTION =======================
    dim3 blocks_row(x_size / (RESULT_STEPS * BLOCKDIM_X), y_size / BLOCKDIM_Y);
    dim3 blocks_col(x_size / BLOCKDIM_X, y_size / (RESULT_STEPS * BLOCKDIM_Y));
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

    float *d_image;
    cudaMalloc((float**) &d_image, bytes);
    cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice);

    float *d_buffer_row, *d_buffer_col;
    cudaMalloc((float**) &d_buffer_row, bytes);
    cudaMalloc((float**) &d_buffer_col, bytes);
    cudaMemset(d_buffer_row, 0, bytes);
    cudaMemset(d_buffer_col, 0, bytes);

    // ====================== VARIABLES FOR DOWNSAMPLING ======================
    dim3 ds_blocks = num_blocks_separable(ds_x_size, ds_y_size, BLOCKDIM_X, BLOCKDIM_Y);

    float *ds_result;
    cudaMalloc((float**) &ds_result, ds_bytes);
    cudaMemset(ds_result, 0, ds_bytes);

    // ===================== CONVOLUTION AND DOWNSAMPLING =====================
    float elapsed_time;
    cudaEvent_t start,stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Convolve and downsample
    convolution_row<<<blocks_row, threads>>>(d_image, d_buffer_row, x_size, y_size);
    convolution_col<<<blocks_col, threads>>>(d_buffer_row, d_buffer_col, x_size, y_size);
    downsample_separable<<<ds_blocks, threads>>>(d_buffer_col, ds_result, x_size, ds_x_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time,start, stop);

    // Copy the result
    cudaMemcpy(result, ds_result, ds_bytes, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_image);
    cudaFree(d_buffer_row);
    cudaFree(d_buffer_col);
    cudaFree(ds_result);

    fprintf(stderr, "TIME: %4.4f\n", elapsed_time);
}
