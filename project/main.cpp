#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "iohandler.h"
#include "scatter.h"

#define DEFAULT_FILENAME "mountains.ppm"

int main(int argc, char **argv) {
    // Read parameters
    char *filename = strdup(DEFAULT_FILENAME);
    if (argc > 1) {
        filename = strdup(argv[1]);
        fprintf(stderr, "Using %s\n", filename);
    }

    // Read image
    int x_size, y_size, maxval;
    unsigned int *image = read_ppm(filename, x_size, y_size, maxval);

    int bytes = x_size * y_size * sizeof(int);
    float *fimage = (float*) mem_check(malloc(bytes));

    // Copy to float image
    for(int i = 0; i < x_size*y_size; i++) {
        fimage[i] = (float) image[i] / 255;
    }

    // Account for downsampling
    int ds_x_size_1 = x_size>>1;
    int ds_y_size_1 = y_size>>1;
    int ds_bytes_1 = ds_x_size_1 * ds_y_size_1 * sizeof(int);

    int ds_x_size_2 = x_size>>2;
    int ds_y_size_2 = y_size>>2;
    int ds_bytes_2 = ds_x_size_2 * ds_y_size_2 * sizeof(int);

    int *result = (int*) mem_check(malloc(ds_bytes_2*5));
    float *fresult = (float*) mem_check(malloc(ds_bytes_2*5));
    //int *result = (int*) mem_check(malloc(ds_bytes_1));
    //float *fresult = (float*) mem_check(malloc(ds_bytes_1));

    // Compute the scattering transform
    scatter(fimage, fresult,
            x_size, y_size, bytes,
            ds_x_size_1, ds_y_size_1, ds_bytes_1,
            ds_x_size_2, ds_y_size_2, ds_bytes_2);
    // scatter_separable(fimage, fresult, x_size, y_size, bytes, ds_x_size, ds_y_size, ds_bytes);

    // Copy to int result
    for(int i = 0; i < ds_x_size_2*ds_y_size_2*5; i++) {
        result[i] = fresult[i] * 255;
    }

    // Write the result
    write_ppm("result.ppm", ds_x_size_2, ds_y_size_2*5, 255, result);

    // Free memory
    free(image);
    free(fimage);
    free(result);
    free(fresult);
}