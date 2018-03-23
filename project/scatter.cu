#include <complex.h>
#include "cuda.h"
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <unistd.h>

#include "scatter.h"

__device__ float convolution_pixel_2D(unsigned int tile[32][32], float filter[7][7], int y, int x) {
    float value = 0;

    // Compute convolution
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            value += tile[y-3+i][x-3+j]*filter[6-i][6-j];
        }
    }

    return value;
}

__global__ void gaussian_convolution(unsigned int *image, int *result, int xsize) {
    // Shared memory tile for image data
    __shared__ unsigned int tile[32][32];

    // Filter with which to convolve the image
    float gaussian[7][7] = {
        {0.00000019425474,  0.000096568274, 0.00010062644,  0.00021978836,  0.00010062644,  0.000096568274, 0.00000019425474},
        {0.0000096568274,	0.00048006195,	0.0050023603,	0.010926159,	0.0050023603,	0.00048006195,	0.0000096568274},
        {0.00010062644,     0.0050023603,	0.052125789,    0.11385319,	    0.052125789,	0.0050023603,	0.00010062644},
        {0.00021978836,     0.010926159,	0.11385319,	    0.24867822,	    0.11385319,	    0.010926159,    0.00021978836},
        {0.00010062644,     0.0050023603,	0.052125789,    0.11385319,	    0.052125789,	0.0050023603,	0.00010062644},
        {0.0000096568274,	0.00048006195,	0.0050023603,	0.010926159,	0.0050023603,	0.00048006195,	0.0000096568274},
        {0.00000019425474,  0.000096568274, 0.00010062644,  0.00021978836,  0.00010062644,  0.000096568274, 0.00000019425474},
    };

    int y = threadIdx.y;
    int x = threadIdx.x;
    int offset = (blockIdx.y*(blockDim.y-6)+y)*xsize + (blockIdx.x*(blockDim.x-6)+x);

    // Load into shared memory
    tile[y][x] = image[offset];
    __syncthreads();

    // Each interior thread computes output
    if (y>2 && y<blockDim.y-3 && x>2 && x<blockDim.x-3) {
        result[offset] = convolution_pixel_2D(tile, gaussian, y, x);
    }
}

__global__ void downsample(int *image, int *dsimage, int xsize, int dsxsize) {
    int yoffset = blockIdx.y*blockDim.y+threadIdx.y;
    int xoffset = blockIdx.x*blockDim.x+threadIdx.x;

    // Save every other pixel in downsampled image
    dsimage[yoffset*dsxsize + xoffset] = image[2*(yoffset*xsize + xoffset)];
}

// ============================================================================
// ============================================================================
// ============================================================================
dim3 numBlocks(int ysize, int xsize, int ythreads, int xthreads) {
    // Compute the number of blocks needed for entire image
    if (ysize % ythreads) {
        ysize = ysize/ythreads*ythreads + ythreads;
    }
    if (xsize % xthreads) {
        xsize = xsize/xthreads*xthreads + xthreads;
    }
    int yblocks = ysize / ythreads;
    int xblocks = xsize / xthreads;

    dim3 blocks(xblocks, yblocks);
    return blocks;
}

void scatter(unsigned int *image, int *result, int xsize, int ysize, int bytes, int dsxsize, int dsysize, int dsbytes) {
    // ====================== VARIABLES FOR CONVOLUTION =======================
    int ythreads = 32;
    int xthreads = 32;
    int yactive = ythreads-6;
    int xactive = xthreads-6;

    dim3 blocks = numBlocks(ysize, xsize, yactive, xactive);
    dim3 threads(xthreads, ythreads);

    // Allocate memory
    unsigned int *dImage;
    cudaMalloc((unsigned int**) &dImage, bytes);
    cudaMemcpy(dImage, image, bytes, cudaMemcpyHostToDevice);

    int *dResult;
    cudaMalloc((int**) &dResult, bytes);
    cudaMemset(dResult, 0, bytes);

    // ====================== VARIABLES FOR DOWNSAMPLING ======================
    dim3 dsblocks = numBlocks(dsysize, dsxsize, ythreads, xthreads);

    // Allocate memory
    int *dsResult;
    cudaMalloc((int**) &dsResult, dsbytes);
    cudaMemset(dsResult, 0, dsbytes);

    // ===================== CONVOLUTION AND DOWNSAMPLING =====================
    gaussian_convolution<<<blocks, threads>>>(dImage, dResult, xsize);
    downsample<<<dsblocks, threads>>>(dResult, dsResult, xsize, dsxsize);
    cudaDeviceSynchronize();

    // Copy the result
    cudaMemcpy(result, dsResult, dsbytes, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(dImage);
    cudaFree(dResult);
    cudaFree(dsResult);
}

