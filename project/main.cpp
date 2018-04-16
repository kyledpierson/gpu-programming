#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "iohandler.h"
#include "scatter.h"

#define DEFAULT_FILENAME "uiuc_sample.ppm"

int main(int argc, char **argv) {
    // VARIABLES
    bool separable = true;

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

    int *result_1 = (int*) mem_check(malloc(ds_bytes_2));
    int *result_2 = (int*) mem_check(malloc(ds_bytes_2));
    int *result_3 = (int*) mem_check(malloc(ds_bytes_2));
    int *result_4 = (int*) mem_check(malloc(ds_bytes_2));
    int *result_5 = (int*) mem_check(malloc(ds_bytes_2));

    float *fresult_1 = (float*) mem_check(malloc(ds_bytes_2));
    float *fresult_2 = (float*) mem_check(malloc(ds_bytes_2));
    float *fresult_3 = (float*) mem_check(malloc(ds_bytes_2));
    float *fresult_4 = (float*) mem_check(malloc(ds_bytes_2));
    float *fresult_5 = (float*) mem_check(malloc(ds_bytes_2));

    // Compute the scattering transform
    scatter(fimage, fresult_1, fresult_2, fresult_3, fresult_4, fresult_5,
            x_size, y_size, bytes,
            ds_x_size_1, ds_y_size_1, ds_bytes_1,
            ds_x_size_2, ds_y_size_2, ds_bytes_2, separable);

    // Copy to int result
    for(int i = 0; i < ds_x_size_2*ds_y_size_2; i++) {
        result_1[i] = fresult_1[i] * 255;
        result_2[i] = fresult_2[i] * 255;
        result_3[i] = fresult_3[i] * 255;
        result_4[i] = fresult_4[i] * 255;
        result_5[i] = fresult_5[i] * 255;
    }

    // Write the result
    write_ppm("result_1.ppm", ds_x_size_2, ds_y_size_2, 255, result_1);
    write_ppm("result_2.ppm", ds_x_size_2, ds_y_size_2, 255, result_2);
    write_ppm("result_3.ppm", ds_x_size_2, ds_y_size_2, 255, result_3);
    write_ppm("result_4.ppm", ds_x_size_2, ds_y_size_2, 255, result_4);
    write_ppm("result_5.ppm", ds_x_size_2, ds_y_size_2, 255, result_5);

    // Free memory
    free(image);
    free(fimage);
    free(result_1);
    free(result_2);
    free(result_3);
    free(result_4);
    free(result_5);
    free(fresult_1);
    free(fresult_2);
    free(fresult_3);
    free(fresult_4);
    free(fresult_5);
}
