#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuComplex.h>

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

// ============================= KERNEL FUNCTIONS =============================
__device__ float convolution_pixel_2D_complex(cuFloatComplex tile[BLOCKDIM_Y][BLOCKDIM_X+1], cuFloatComplex filter[KERNEL_SIZE][KERNEL_SIZE], int x, int y) {
    cuFloatComplex value = make_cuFloatComplex(0, 0);

    // Compute convolution
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            value = cuCaddf(value, cuCmulf(tile[y-HALO_SIZE+i][x-HALO_SIZE+j], filter[KERNEL_SIZE-i-1][KERNEL_SIZE-j-1]));
        }
    }
    return cuCabsf(value);
}

__device__ float convolution_pixel_2D(float tile[BLOCKDIM_Y][BLOCKDIM_X+1], float filter[KERNEL_SIZE][KERNEL_SIZE], int x, int y) {
    float value = 0;

    // Compute convolution
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            value += tile[y-HALO_SIZE+i][x-HALO_SIZE+j]*filter[KERNEL_SIZE-i-1][KERNEL_SIZE-j-1];
        }
    }
    return value;
}

__global__ void gaussian_convolution_2D(float *image, float *result, int x_size, int ds_x_size) {
    float gaussian_2D[7][7] = {
        {0.00000019, 0.00009657, 0.00010063, 0.00021979, 0.00010063, 0.00009657, 0.00000019},
        {0.00000966, 0.00048006, 0.00500236, 0.01092616, 0.00500236, 0.00048006, 0.00000966},
        {0.00010063, 0.00500236, 0.05212579, 0.11385319, 0.05212579, 0.00500236, 0.00010063},
        {0.00021979, 0.01092616, 0.11385319, 0.24867822, 0.11385319, 0.01092616, 0.00021979},
        {0.00010063, 0.00500236, 0.05212579, 0.11385319, 0.05212579, 0.00500236, 0.00010063},
        {0.00000966, 0.00048006, 0.00500236, 0.01092616, 0.00500236, 0.00048006, 0.00000966},
        {0.00000019, 0.00009657, 0.00010063, 0.00021979, 0.00010063, 0.00009657, 0.00000019},
    };

    // Shared memory tile for image data
    __shared__ float tile[BLOCKDIM_Y][BLOCKDIM_X+1];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int x_offset = blockIdx.x*(blockDim.x-(2*HALO_SIZE))+x;
    int y_offset = blockIdx.y*(blockDim.y-(2*HALO_SIZE))+y;

    // Load into shared memory
    tile[y][x] = image[y_offset*x_size + x_offset];
    __syncthreads();

    // Each interior thread computes output
    if (x>=HALO_SIZE && x<blockDim.x-HALO_SIZE && y>=HALO_SIZE && y<blockDim.y-HALO_SIZE) {
        result[(y_offset/2)*ds_x_size + (x_offset/2)] = 2*convolution_pixel_2D(tile, gaussian_2D, x, y);
    }
}

__global__ void morlet_1_convolution_2D(float *image, float *result, int x_size) {
    cuFloatComplex a = make_cuFloatComplex( 0,           0         );
    cuFloatComplex b = make_cuFloatComplex(-0.00000003,  0.00000016);
    cuFloatComplex c = make_cuFloatComplex(-0.00000150, -0.00000120);
    cuFloatComplex d = make_cuFloatComplex( 0.00000305,  0         );
    cuFloatComplex e = make_cuFloatComplex( 0.00002050, -0.00002731);
    cuFloatComplex f = make_cuFloatComplex(-0.00033877,  0.00192026);
    cuFloatComplex g = make_cuFloatComplex(-0.01767897, -0.01414889);
    cuFloatComplex h = make_cuFloatComplex( 0.03599448,  0         );
    cuFloatComplex i = make_cuFloatComplex( 0.00046656, -0.00062166);
    cuFloatComplex j = make_cuFloatComplex(-0.00771040,  0.04370487);
    cuFloatComplex k = make_cuFloatComplex(-0.40237140, -0.32202720);
    cuFloatComplex l = make_cuFloatComplex( 0.81923060,  0         );

    cuFloatComplex morlet_2D_1[7][7] = {
        {a, a, a, a, a, a, a},
        {a, b, c, d, c, b, a},
        {e, f, g, h, g, f, e},
        {i, j, k, l, k, j, i},
        {e, f, g, h, g, f, e},
        {a, b, c, d, c, b, a},
        {a, a, a, a, a, a, a}
    };

    // Shared memory tile for image data
    __shared__ cuFloatComplex tile[BLOCKDIM_Y][BLOCKDIM_X+1];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int offset = (blockIdx.y*(blockDim.y-(2*HALO_SIZE))+y)*x_size + (blockIdx.x*(blockDim.x-(2*HALO_SIZE))+x);

    // Load into shared memory
    tile[y][x] = make_cuFloatComplex(image[offset], 0);
    __syncthreads();

    // Each interior thread computes output
    if (x>=HALO_SIZE && x<blockDim.x-HALO_SIZE && y>=HALO_SIZE && y<blockDim.y-HALO_SIZE) {
        result[offset] = convolution_pixel_2D_complex(tile, morlet_2D_1, x, y);
    }
}

__global__ void morlet_2_convolution_2D(float *image, float *result, int x_size) {
    cuFloatComplex a = make_cuFloatComplex( 0,           0         );
    cuFloatComplex b = make_cuFloatComplex(-0.00000003,  0.00000016);
    cuFloatComplex c = make_cuFloatComplex(-0.00000150, -0.00000120);
    cuFloatComplex d = make_cuFloatComplex( 0.00000305,  0         );
    cuFloatComplex e = make_cuFloatComplex( 0.00002050, -0.00002731);
    cuFloatComplex f = make_cuFloatComplex(-0.00033877,  0.00192026);
    cuFloatComplex g = make_cuFloatComplex(-0.01767897, -0.01414889);
    cuFloatComplex h = make_cuFloatComplex( 0.03599448,  0         );
    cuFloatComplex i = make_cuFloatComplex( 0.00046656, -0.00062166);
    cuFloatComplex j = make_cuFloatComplex(-0.00771040,  0.04370487);
    cuFloatComplex k = make_cuFloatComplex(-0.40237140, -0.32202720);
    cuFloatComplex l = make_cuFloatComplex( 0.81923060,  0         );

    cuFloatComplex morlet_2D_2[7][7] = {
        {a, a, e, i, e, a, a},
        {a, b, f, j, f, b, a},
        {a, c, g, k, g, c, a},
        {a, d, h, l, h, d, a},
        {a, c, g, k, g, c, a},
        {a, b, f, j, f, b, a},
        {a, a, e, i, e, a, a}
    };

    // Shared memory tile for image data
    __shared__ cuFloatComplex tile[BLOCKDIM_Y][BLOCKDIM_X+1];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int offset = (blockIdx.y*(blockDim.y-(2*HALO_SIZE))+y)*x_size + (blockIdx.x*(blockDim.x-(2*HALO_SIZE))+x);

    // Load into shared memory
    tile[y][x] = make_cuFloatComplex(image[offset], 0);
    __syncthreads();

    // Each interior thread computes output
    if (x>=HALO_SIZE && x<blockDim.x-HALO_SIZE && y>=HALO_SIZE && y<blockDim.y-HALO_SIZE) {
        result[offset] = convolution_pixel_2D_complex(tile, morlet_2D_2, x, y);
    }
}

// ============================================================================
// ============================================================================
// ============================================================================
void scatter(float *image, float *result,
             int x_size, int y_size, int bytes,
             int ds_x_size_1, int ds_y_size_1, int ds_bytes_1,
             int ds_x_size_2, int ds_y_size_2, int ds_bytes_2) {
    int x_active = BLOCKDIM_X-(2*HALO_SIZE);
    int y_active = BLOCKDIM_Y-(2*HALO_SIZE);

    dim3 blocks = num_blocks(x_size, y_size, x_active, y_active);
    dim3 ds_blocks = num_blocks(ds_x_size_1, ds_y_size_1, x_active, y_active);
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

    // Allocate memory
    float *d_image;
    cudaMalloc((float**) &d_image, bytes);
    cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice);

    // Layer 1 - low pass
    float *lp_1, *lp_2;
    cudaMalloc((float**) &lp_1, ds_bytes_1);
    cudaMalloc((float**) &lp_2, ds_bytes_2);
    cudaMemset(lp_1, 0, ds_bytes_1);
    cudaMemset(lp_2, 0, ds_bytes_2);

    // Layer 1 - high pass
    float *hp_1, *hp_2, *hp_3, *hp_4;
    cudaMalloc((float**) &hp_1, bytes);
    cudaMalloc((float**) &hp_2, bytes);
    cudaMalloc((float**) &hp_3, ds_bytes_1);
    cudaMalloc((float**) &hp_4, ds_bytes_1);
    cudaMemset(hp_1, 0, bytes);
    cudaMemset(hp_2, 0, bytes);
    cudaMemset(hp_3, 0, ds_bytes_1);
    cudaMemset(hp_4, 0, ds_bytes_1);

    // Layer 2 - low pass
    float *lp_3, *lp_4, *lp_5, *lp_6, *lp_7, *lp_8;
    cudaMalloc((float**) &lp_3, ds_bytes_1);
    cudaMalloc((float**) &lp_4, ds_bytes_2);
    cudaMalloc((float**) &lp_5, ds_bytes_1);
    cudaMalloc((float**) &lp_6, ds_bytes_2);
    cudaMalloc((float**) &lp_7, ds_bytes_2);
    cudaMalloc((float**) &lp_8, ds_bytes_2);
    cudaMemset(lp_3, 0, ds_bytes_1);
    cudaMemset(lp_4, 0, ds_bytes_2);
    cudaMemset(lp_5, 0, ds_bytes_1);
    cudaMemset(lp_6, 0, ds_bytes_2);
    cudaMemset(lp_7, 0, ds_bytes_2);
    cudaMemset(lp_8, 0, ds_bytes_2);

    // ========================= SCATTERING TRANSFORM =========================
    float elapsed_time;
    cudaEvent_t start,stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // ========================================================================
    // Layer 1 - low pass
    gaussian_convolution_2D<<<blocks, threads>>>(d_image, lp_1, x_size, ds_x_size_1);
    gaussian_convolution_2D<<<ds_blocks, threads>>>(lp_1, lp_2, ds_x_size_1, ds_x_size_2);

    // Layer 1 - high pass
    morlet_1_convolution_2D<<<blocks, threads>>>(d_image, hp_1, x_size);
    morlet_2_convolution_2D<<<blocks, threads>>>(d_image, hp_2, x_size);
    morlet_1_convolution_2D<<<ds_blocks, threads>>>(lp_1, hp_3, ds_x_size_1);
    morlet_2_convolution_2D<<<ds_blocks, threads>>>(lp_1, hp_4, ds_x_size_1);

    // Layer 2 - low pass
    gaussian_convolution_2D<<<blocks, threads>>>(hp_1, lp_3, x_size, ds_x_size_1);
    gaussian_convolution_2D<<<ds_blocks, threads>>>(lp_3, lp_4, ds_x_size_1, ds_x_size_2);
    gaussian_convolution_2D<<<blocks, threads>>>(hp_2, lp_5, x_size, ds_x_size_1);
    gaussian_convolution_2D<<<ds_blocks, threads>>>(lp_5, lp_6, ds_x_size_1, ds_x_size_2);
    gaussian_convolution_2D<<<ds_blocks, threads>>>(hp_3, lp_7, ds_x_size_1, ds_x_size_2);
    gaussian_convolution_2D<<<ds_blocks, threads>>>(hp_4, lp_8, ds_x_size_1, ds_x_size_2);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time,start, stop);
    // ========================================================================

    // Copy the result
    int offset = ds_x_size_2*ds_y_size_2;
    cudaMemcpy(result, lp_2, ds_bytes_2, cudaMemcpyDeviceToHost);
    cudaMemcpy(result+offset, lp_4, ds_bytes_2, cudaMemcpyDeviceToHost);
    cudaMemcpy(result+(offset*2), lp_6, ds_bytes_2, cudaMemcpyDeviceToHost);
    cudaMemcpy(result+(offset*3), lp_7, ds_bytes_2, cudaMemcpyDeviceToHost);
    cudaMemcpy(result+(offset*4), lp_8, ds_bytes_2, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_image);
    cudaFree(lp_1);
    cudaFree(lp_2);
    cudaFree(hp_1);
    cudaFree(hp_2);
    cudaFree(hp_3);
    cudaFree(hp_4);
    cudaFree(lp_3);
    cudaFree(lp_4);
    cudaFree(lp_5);
    cudaFree(lp_6);
    cudaFree(lp_7);
    cudaFree(lp_8);

    fprintf(stderr, "TIME: %4.4f\n", elapsed_time);
}

