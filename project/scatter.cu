#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>

#include "scatter.h"
#include "iohandler.h"
#include "Log.h"

// ============================= HELPER FUNCTIONS =============================
__constant__ float d_kernel[KERNEL_SIZE];
void copy_kernel_1D(float h_kernel[KERNEL_SIZE]) {
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float));
}

dim3 num_blocks(int x_size, int y_size, int x_threads, int y_threads) {
    // Compute the number of blocks needed for entire image
    int x_blocks = x_size / x_threads;
    int y_blocks = y_size / y_threads;

    dim3 blocks(x_blocks, y_blocks);
    return blocks;
}

// ============================= DEVICE FUNCTIONS =============================
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

// ============================= KERNEL FUNCTIONS =============================
__global__ void multiply(cuComplex *image, float *filter, int x_size) {
    int x = blockIdx.x * BLOCKDIM_X + threadIdx.x;
    int y = blockIdx.y * BLOCKDIM_Y + threadIdx.y;
    int offset = y*x_size + x;

    image[offset] = cuCmulf(image[offset], make_cuFloatComplex(filter[offset], 0));
}

__global__ void gaussian_convolution_2D(float *image, float *result, int x_size, int ds_x_size) {
    float gaussian_2D[7][7] = {
        {0.004922330, 0.009196123, 0.013380281, 0.015161844, 0.013380281, 0.009196123, 0.004922330},
        {0.009196123, 0.017180620, 0.024997653, 0.028326053, 0.024997653, 0.017180620, 0.009196123},
        {0.013380281, 0.024997653, 0.036371373, 0.041214164, 0.036371373, 0.024997653, 0.013380281},
        {0.015161844, 0.028326053, 0.041214164, 0.046701763, 0.041214164, 0.028326053, 0.015161844},
        {0.013380281, 0.024997653, 0.036371373, 0.041214164, 0.036371373, 0.024997653, 0.013380281},
        {0.009196123, 0.017180620, 0.024997653, 0.028326053, 0.024997653, 0.017180620, 0.009196123},
        {0.004922330, 0.009196123, 0.013380281, 0.015161844, 0.013380281, 0.009196123, 0.004922330},
    };

    // Shared memory tile for image data
    __shared__ float tile[BLOCKDIM_Y][BLOCKDIM_X+1];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int x_offset = blockIdx.x*(BLOCKDIM_X-(2*HALO_SIZE))+x;
    int y_offset = blockIdx.y*(BLOCKDIM_Y-(2*HALO_SIZE))+y;

    // Load into shared memory
    tile[y][x] = image[y_offset*x_size + x_offset];
    __syncthreads();

    // Each interior thread computes output
    if (x>=HALO_SIZE && x<BLOCKDIM_X-HALO_SIZE && y>=HALO_SIZE && y<BLOCKDIM_Y-HALO_SIZE) {
        result[(y_offset/2)*ds_x_size + (x_offset/2)] = 2*convolution_pixel_2D(tile, gaussian_2D, x, y);
    }
}

__global__ void morlet_1_convolution_2D(float *image, float *result, int x_size) {
    cuFloatComplex a = make_cuFloatComplex(0.000379696, -0.000405881);
    cuFloatComplex b = make_cuFloatComplex(-0.0000489192, 0.001072378);
    cuFloatComplex c = make_cuFloatComplex(-0.001174476, -0.001103299);
    cuFloatComplex d = make_cuFloatComplex(0.001687397, 0);

    cuFloatComplex e = make_cuFloatComplex(0.004625649, -0.004944642);
    cuFloatComplex f = make_cuFloatComplex(-0.000595958, 0.01306423);
    cuFloatComplex g = make_cuFloatComplex(-0.01430805, -0.01344093);
    cuFloatComplex h = make_cuFloatComplex(0.02055671, 0);

    cuFloatComplex i = make_cuFloatComplex(0.02073072, -0.02216035);
    cuFloatComplex j = make_cuFloatComplex(-0.0026709, 0.05854983);
    cuFloatComplex k = make_cuFloatComplex(-0.06412421, -0.06023807);
    cuFloatComplex l = make_cuFloatComplex(0.09212878, 0);

    cuFloatComplex m = make_cuFloatComplex(0.03417918, -0.03653624);
    cuFloatComplex n = make_cuFloatComplex(-0.00440357, 0.09653235);
    cuFloatComplex o = make_cuFloatComplex(-0.1057229, -0.09931579);
    cuFloatComplex p = make_cuFloatComplex(0.1518947, 0);

    cuFloatComplex ac = cuConjf(a);
    cuFloatComplex bc = cuConjf(b);
    cuFloatComplex cc = cuConjf(c);

    cuFloatComplex ec = cuConjf(e);
    cuFloatComplex fc = cuConjf(f);
    cuFloatComplex gc = cuConjf(g);

    cuFloatComplex ic = cuConjf(i);
    cuFloatComplex jc = cuConjf(j);
    cuFloatComplex kc = cuConjf(k);

    cuFloatComplex mc = cuConjf(m);
    cuFloatComplex nc = cuConjf(n);
    cuFloatComplex oc = cuConjf(o);

    cuFloatComplex morlet_2D_1[7][7] = {
        {a, b, c, d, cc, bc, ac},
        {e, f, g, h, gc, fc, ec},
        {i, j, k, l, kc, jc, ic},
        {m, n, o, p, oc, nc, mc},
        {i, j, k, l, kc, jc, ic},
        {e, f, g, h, gc, fc, ec},
        {a, b, c, d, cc, bc, ac}
    };

    // Shared memory tile for image data
    __shared__ cuFloatComplex tile[BLOCKDIM_Y][BLOCKDIM_X+1];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int offset = (blockIdx.y*(BLOCKDIM_Y-(2*HALO_SIZE))+y)*x_size + (blockIdx.x*(BLOCKDIM_X-(2*HALO_SIZE))+x);

    // Load into shared memory
    tile[y][x] = make_cuFloatComplex(image[offset], 0);
    __syncthreads();

    // Each interior thread computes output
    if (x>=HALO_SIZE && x<BLOCKDIM_X-HALO_SIZE && y>=HALO_SIZE && y<BLOCKDIM_Y-HALO_SIZE) {
        result[offset] = convolution_pixel_2D_complex(tile, morlet_2D_1, x, y);
    }
}

__global__ void morlet_2_convolution_2D(float *image, float *result, int x_size) {
    cuFloatComplex a = make_cuFloatComplex(0.000379696, -0.000405881);
    cuFloatComplex b = make_cuFloatComplex(-0.0000489192, 0.001072378);
    cuFloatComplex c = make_cuFloatComplex(-0.001174476, -0.001103299);
    cuFloatComplex d = make_cuFloatComplex(0.001687397, 0);

    cuFloatComplex e = make_cuFloatComplex(0.004625649, -0.004944642);
    cuFloatComplex f = make_cuFloatComplex(-0.000595958, 0.01306423);
    cuFloatComplex g = make_cuFloatComplex(-0.01430805, -0.01344093);
    cuFloatComplex h = make_cuFloatComplex(0.02055671, 0);

    cuFloatComplex i = make_cuFloatComplex(0.02073072, -0.02216035);
    cuFloatComplex j = make_cuFloatComplex(-0.0026709, 0.05854983);
    cuFloatComplex k = make_cuFloatComplex(-0.06412421, -0.06023807);
    cuFloatComplex l = make_cuFloatComplex(0.09212878, 0);

    cuFloatComplex m = make_cuFloatComplex(0.03417918, -0.03653624);
    cuFloatComplex n = make_cuFloatComplex(-0.00440357, 0.09653235);
    cuFloatComplex o = make_cuFloatComplex(-0.1057229, -0.09931579);
    cuFloatComplex p = make_cuFloatComplex(0.1518947, 0);

    cuFloatComplex ac = cuConjf(a);
    cuFloatComplex bc = cuConjf(b);
    cuFloatComplex cc = cuConjf(c);

    cuFloatComplex ec = cuConjf(e);
    cuFloatComplex fc = cuConjf(f);
    cuFloatComplex gc = cuConjf(g);

    cuFloatComplex ic = cuConjf(i);
    cuFloatComplex jc = cuConjf(j);
    cuFloatComplex kc = cuConjf(k);

    cuFloatComplex mc = cuConjf(m);
    cuFloatComplex nc = cuConjf(n);
    cuFloatComplex oc = cuConjf(o);

    cuFloatComplex morlet_2D_2[7][7] = {
        {a, e, i, m, i, e, a},
        {b, f, j, n, j, f, b},
        {c, g, k, o, k, g, c},
        {d, h, l, p, l, h, d},
        {cc, gc, kc, oc, kc, gc, cc},
        {bc, fc, jc, nc, jc, fc, bc},
        {ac, ec, ic, mc, ic, ec, ac}
    };

    // Shared memory tile for image data
    __shared__ cuFloatComplex tile[BLOCKDIM_Y][BLOCKDIM_X+1];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int offset = (blockIdx.y*(BLOCKDIM_Y-(2*HALO_SIZE))+y)*x_size + (blockIdx.x*(BLOCKDIM_X-(2*HALO_SIZE))+x);

    // Load into shared memory
    tile[y][x] = make_cuFloatComplex(image[offset], 0);
    __syncthreads();

    // Each interior thread computes output
    if (x>=HALO_SIZE && x<BLOCKDIM_X-HALO_SIZE && y>=HALO_SIZE && y<BLOCKDIM_Y-HALO_SIZE) {
        result[offset] = convolution_pixel_2D_complex(tile, morlet_2D_2, x, y);
    }
}

__global__ void gaussian_convolution_row(float *image, float *result, int x_size, int y_size) {
    __shared__ float tile[BLOCKDIM_Y][(RESULT_STEPS + 2*HALO_STEPS) * BLOCKDIM_X];

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

__global__ void gaussian_convolution_col(float *image, float *result, int x_size, int y_size) {
    __shared__ float tile[BLOCKDIM_X][(RESULT_STEPS + 2*HALO_STEPS) * BLOCKDIM_Y + 1];

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

__global__ void downsample(float *image, float *ds_image, int x_size, int ds_x_size) {
    int x_offset = blockIdx.x*BLOCKDIM_X+threadIdx.x;
    int y_offset = blockIdx.y*BLOCKDIM_Y+threadIdx.y;

    // Save every other pixel in downsampled image
    ds_image[y_offset*ds_x_size + x_offset] = 2*image[2*(y_offset*x_size + x_offset)];
}

// ============================================================================
// ============================================================================
// ============================================================================
void gaussian_convolution_1D(Job* job, cudaStream_t stream, float* d_image, float* d_result, int x_size, int y_size, int bytes, int ds_x_size, int ds_y_size, int ds_bytes) {
    dim3 blocks_row(x_size / (RESULT_STEPS * BLOCKDIM_X), y_size / BLOCKDIM_Y);
    dim3 blocks_col(x_size / BLOCKDIM_X, y_size / (RESULT_STEPS * BLOCKDIM_Y));
    dim3 ds_blocks = num_blocks(ds_x_size, ds_y_size, BLOCKDIM_X, BLOCKDIM_Y);
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

    float *d_buffer_row, *d_buffer_col;
    cudaMalloc((float**) &d_buffer_row, bytes);
    cudaMalloc((float**) &d_buffer_col, bytes);
    cudaMemset(d_buffer_row, 0, bytes);
    cudaMemset(d_buffer_col, 0, bytes);

    gaussian_convolution_row<<<blocks_row, threads,0,stream>>>(d_image, d_buffer_row, x_size, y_size);
    gaussian_convolution_col<<<blocks_col, threads,0,stream>>>(d_buffer_row, d_buffer_col, x_size, y_size);
    downsample<<<ds_blocks, threads,0,stream>>>(d_buffer_col, d_result, x_size, ds_x_size);

    //Might want to re-work this
    job->addFree(d_buffer_row,true);
    job->addFree(d_buffer_col,true);
}

void initConsts() {
    float gaussian_1D[7] = {0.071303, 0.131514, 0.189879, 0.214607, 0.189879, 0.131514, 0.071303};
    copy_kernel_1D(gaussian_1D);
}

void scatter(float *image, JobScheduler* scheduler, const std::string& outputFile,
             int x_size, int y_size, int bytes,
             int ds_x_size_1, int ds_y_size_1, int ds_bytes_1,
             int ds_x_size_2, int ds_y_size_2, int ds_bytes_2, bool fourier, bool separable) {

    int x_active = BLOCKDIM_X-(2*HALO_SIZE);
    int y_active = BLOCKDIM_Y-(2*HALO_SIZE);

    uint64_t totalRequiredMemory = 0;
    totalRequiredMemory += (ds_bytes_1*5) + (ds_bytes_2*5) + (bytes*3);
    Job* job = scheduler->addJob();
    auto lambda = [=] (cudaStream_t stream) {
        //printf("Executing job lambda...\n");

        dim3 blocks = num_blocks(x_size, y_size, x_active, y_active);
        dim3 ds_blocks = num_blocks(ds_x_size_1, ds_y_size_1, x_active, y_active);
        dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

        // Variables
        float *d_image, *lp_1, *lp_2, *lp_3, *lp_4, *lp_5, *lp_6, *lp_7, *lp_8, *hp_1, *hp_2, *hp_3, *hp_4;

        // ----------------------------------------------------------------------------------------------------
        cudaMalloc((float**) &d_image, bytes);
        cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice);

        cudaMalloc((float**) &lp_1, ds_bytes_1);
        cudaMalloc((float**) &hp_1, bytes);
        cudaMalloc((float**) &hp_2, bytes);
        cudaMemset(lp_1, 0, ds_bytes_1);
        cudaMemset(hp_1, 0, bytes);
        cudaMemset(hp_2, 0, bytes);

        if (fourier) {
            cufftHandle plan_r2c, plan_c2r;
            cufftPlan2d(&plan_r2c, y_size, x_size, CUFFT_R2C);
            cufftPlan2d(&plan_c2r, y_size, x_size, CUFFT_C2R);

            // Create complex image on device
            cuComplex *c_image, *dc_image;
            int c_bytes = x_size * y_size * sizeof(cuComplex);
            cudaMalloc((cuComplex**) &c_image, c_bytes);
            cudaMalloc((cuComplex**) &dc_image, c_bytes);

            // Convert the image to the Fourier domain
            cufftExecR2C(plan_r2c, d_image, c_image);

            // Read the gaussian filter (Fourier domain) ==========================================
            read_filter("gaussian_480_640.txt", image);
            cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice);

            // Perform multiplication in the Fourier domain
            cudaMemcpy(dc_image, c_image, bytes, cudaMemcpyDeviceToDevice);
            multiply<<<blocks, threads, 0, stream>>>(dc_image, d_image, x_size);

            // Convert the image back to the spatial domain and downsample
            cufftExecC2R(plan_c2r, dc_image, d_image);
            downsample<<<ds_blocks, threads>>>(d_image, lp_1, x_size, ds_x_size_1);

            // Read the morlet 1 filter (Fourier domain) ==========================================
            read_filter("morlet_1_480_640.txt", image);
            cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice);

            // Perform multiplication in the Fourier domain
            cudaMemcpy(dc_image, c_image, bytes, cudaMemcpyDeviceToDevice);
            multiply<<<blocks, threads, 0, stream>>>(dc_image, d_image, x_size);

            // Convert the image back to the spatial domain and downsample
            cufftExecC2R(plan_c2r, dc_image, hp_1);

            // Read the morlet 2 filter (Fourier domain) ==========================================
            read_filter("morlet_2_480_640.txt", image);
            cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice);

            // Perform multiplication in the Fourier domain
            cudaMemcpy(dc_image, c_image, bytes, cudaMemcpyDeviceToDevice);
            multiply<<<blocks, threads, 0, stream>>>(dc_image, d_image, x_size);

            // Convert the image back to the spatial domain and downsample
            cufftExecC2R(plan_c2r, dc_image, hp_2);

            // Free memory
            cudaFree(c_image);
            cudaFree(dc_image);
        } else {
            if (separable) {
                gaussian_convolution_1D(job,stream,d_image, lp_1, x_size, y_size, bytes, ds_x_size_1, ds_y_size_1, ds_bytes_1);
            } else {
                gaussian_convolution_2D<<<blocks, threads,0,stream>>>(d_image, lp_1, x_size, ds_x_size_1);
            }
            morlet_1_convolution_2D<<<blocks, threads,0,stream>>>(d_image, hp_1, x_size);
            morlet_2_convolution_2D<<<blocks, threads,0,stream>>>(d_image, hp_2, x_size);
        }
        free(image);
        cudaFree(d_image);

        // ----------------------------------------------------------------------------------------------------
        cudaMalloc((float**) &lp_3, ds_bytes_1);
        cudaMemset(lp_3, 0, ds_bytes_1);
        if (separable) {
            gaussian_convolution_1D(job,stream,hp_1, lp_3, x_size, y_size, bytes, ds_x_size_1, ds_y_size_1, ds_bytes_1);
        } else {
            gaussian_convolution_2D<<<blocks, threads,0,stream>>>(hp_1, lp_3, x_size, ds_x_size_1);
        }
        cudaFree(hp_1);

        // ----------------------------------------------------------------------------------------------------
        cudaMalloc((float**) &lp_5, ds_bytes_1);
        cudaMemset(lp_5, 0, ds_bytes_1);
        if (separable) {
            gaussian_convolution_1D(job,stream,hp_2, lp_5, x_size, y_size, bytes, ds_x_size_1, ds_y_size_1, ds_bytes_1);
        } else {
            gaussian_convolution_2D<<<blocks, threads,0,stream>>>(hp_2, lp_5, x_size, ds_x_size_1);
        }
        cudaFree(hp_2);

        // ----------------------------------------------------------------------------------------------------
        cudaMalloc((float**) &lp_2, ds_bytes_2);
        cudaMalloc((float**) &hp_3, ds_bytes_1);
        cudaMalloc((float**) &hp_4, ds_bytes_1);
        cudaMemset(lp_2, 0, ds_bytes_2);
        cudaMemset(hp_3, 0, ds_bytes_1);
        cudaMemset(hp_4, 0, ds_bytes_1);
        if (separable) {
            gaussian_convolution_1D(job,stream,lp_1, lp_2, ds_x_size_1, ds_y_size_1, ds_bytes_1, ds_x_size_2, ds_y_size_2, ds_bytes_2);
        } else {
            gaussian_convolution_2D<<<ds_blocks, threads,0,stream>>>(lp_1, lp_2, ds_x_size_1, ds_x_size_2);
        }
        morlet_1_convolution_2D<<<ds_blocks, threads,0,stream>>>(lp_1, hp_3, ds_x_size_1);
        morlet_2_convolution_2D<<<ds_blocks, threads,0,stream>>>(lp_1, hp_4, ds_x_size_1);
        cudaFree(lp_1);

        // ----------------------------------------------------------------------------------------------------
        cudaMalloc((float**) &lp_4, ds_bytes_2);
        cudaMemset(lp_4, 0, ds_bytes_2);
        if (separable) {
            gaussian_convolution_1D(job,stream,lp_3, lp_4, ds_x_size_1, ds_y_size_1, ds_bytes_1, ds_x_size_2, ds_y_size_2, ds_bytes_2);
        } else {
            gaussian_convolution_2D<<<ds_blocks, threads,0,stream>>>(lp_3, lp_4, ds_x_size_1, ds_x_size_2);
        }
        cudaFree(lp_3);

        // ----------------------------------------------------------------------------------------------------
        cudaMalloc((float**) &lp_6, ds_bytes_2);
        cudaMemset(lp_6, 0, ds_bytes_2);
        if (separable) {
            gaussian_convolution_1D(job,stream,lp_5, lp_6, ds_x_size_1, ds_y_size_1, ds_bytes_1, ds_x_size_2, ds_y_size_2, ds_bytes_2);
        } else {
            gaussian_convolution_2D<<<ds_blocks, threads,0,stream>>>(lp_5, lp_6, ds_x_size_1, ds_x_size_2);
        }
        cudaFree(lp_5);

        // ----------------------------------------------------------------------------------------------------
        cudaMalloc((float**) &lp_7, ds_bytes_2);
        cudaMemset(lp_7, 0, ds_bytes_2);
        if (separable) {
            gaussian_convolution_1D(job,stream,hp_3, lp_7, ds_x_size_1, ds_y_size_1, ds_bytes_1, ds_x_size_2, ds_y_size_2, ds_bytes_2);
        } else {
            gaussian_convolution_2D<<<ds_blocks, threads,0,stream>>>(hp_3, lp_7, ds_x_size_1, ds_x_size_2);
        }
        cudaFree(hp_3);

        // ----------------------------------------------------------------------------------------------------
        cudaMalloc((float**) &lp_8, ds_bytes_2);
        cudaMemset(lp_8, 0, ds_bytes_2);
        if (separable) {
            gaussian_convolution_1D(job,stream,hp_4, lp_8, ds_x_size_1, ds_y_size_1, ds_bytes_1, ds_x_size_2, ds_y_size_2, ds_bytes_2);
        } else {
            gaussian_convolution_2D<<<ds_blocks, threads,0,stream>>>(hp_4, lp_8, ds_x_size_1, ds_x_size_2);
        }
        cudaFree(hp_4);

        // ========================================================================
        job->registerCleanup([=] () {
            //printf("Executing cleanup\n");
            int *iresult = (int*) mem_check(malloc(ds_bytes_2*5));
            float *result = (float*) mem_check(malloc(ds_bytes_2*5));
            int offset = ds_x_size_2*ds_y_size_2;

            cudaMemcpy(result, lp_2, ds_bytes_2, cudaMemcpyDeviceToHost);
            cudaMemcpy(result+offset, lp_4, ds_bytes_2, cudaMemcpyDeviceToHost);
            cudaMemcpy(result+2*offset, lp_6, ds_bytes_2, cudaMemcpyDeviceToHost);
            cudaMemcpy(result+3*offset, lp_7, ds_bytes_2, cudaMemcpyDeviceToHost);
            cudaMemcpy(result+4*offset, lp_8, ds_bytes_2, cudaMemcpyDeviceToHost);

            // Find the max for each image
            float maxval_1 = 0;
            float maxval_2 = 0;
            float maxval_3 = 0;
            float maxval_4 = 0;
            float maxval_5 = 0;
            for(int i = 0; i < offset*5; i++) {
                if (i/offset == 0 && result[i] > maxval_1) {
                    maxval_1 = result[i];
                } else if (i/offset == 1 && result[i] > maxval_2) {
                    maxval_2 = result[i];
                } else if (i/offset == 2 && result[i] > maxval_3) {
                    maxval_3 = result[i];
                } else if (i/offset == 3 && result[i] > maxval_4) {
                    maxval_4 = result[i];
                } else if (i/offset == 4 && result[i] > maxval_5) {
                    maxval_5 = result[i];
                }
            }

            // Re-normalize each image to a scale of 0-255
            for(int i = 0; i < offset*5; i++) {
                if (i/offset == 0) {
                    iresult[i] = (result[i] / maxval_1) * 255;
                } else if (i/offset == 1) {
                    iresult[i] = (result[i] / maxval_2) * 255;
                } else if (i/offset == 2) {
                    iresult[i] = (result[i] / maxval_3) * 255;
                } else if (i/offset == 3) {
                    iresult[i] = (result[i] / maxval_4) * 255;
                } else if (i/offset == 4) {
                    iresult[i] = (result[i] / maxval_5) * 255;
                }
            }
            LOG_DEBUG(std::string("Writing to output file: ") + outputFile);
            write_ppm((char*)outputFile.c_str(), ds_x_size_2, ds_y_size_2*5, 255, iresult);

            job->FreeMemory();

            // Free memory
            free(result);
            free(iresult);
            cudaFree(lp_2);
            cudaFree(lp_4);
            cudaFree(lp_6);
            cudaFree(lp_7);
            cudaFree(lp_8);
            LOG_DEBUG("Cleanup complete");
        });
        job->setDone(); // do this when you're ready to call your cleanup
        cudaStreamAddCallback(stream,&Job::cudaCb,(void*)job,0);
    };

    job->addStage(lambda,totalRequiredMemory,bytes);
    job->queue();
}


