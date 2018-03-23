#include <complex.h>
#include "cuda.h"
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <unistd.h>

#include "scatter.h"

__device__ float convolution_pixel_1D(p0, p1, p2, p3, p4, p5, p6, float filter[7]) {
    return p0*filter[0] + p1*filter[1] + p2*filter[2] + p3*filter[3] + p4*filter[4] + p5*filter[5] + p6*filter[6];
}

__global__ void gaussian_convolution_separable(unsigned int *image, int *result, int xsize) {
    float gaussian[] = {0.000395, 0.021639, 0.229031, 0.497871, 0.229031, 0.021639, 0.000395};
}
