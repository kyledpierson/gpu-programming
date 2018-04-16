#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "iohandler.h"
#include "scatter.h"
#include "JobScheduler.h"
#include "test.h"

#define DEFAULT_FILENAME "uiuc_sample.ppm"

void scheduleForTransformation(JobScheduler* scheduler, char* inputFile, char* outputFile)
{
    bool separable = false;
    int x_size, y_size, maxval;
    unsigned int *image = read_ppm(inputFile, x_size, y_size, maxval);
    int bytes = x_size * y_size * sizeof(int);
    float *fimage = (float*) mem_check(malloc(bytes));

    // Copy to float image
    for(int i = 0; i < x_size*y_size; i++) {
        fimage[i] = (float) image[i] / 255;
    }
    free(image);

    // Account for downsampling
    int ds_x_size_1 = x_size>>1;
    int ds_y_size_1 = y_size>>1;
    int ds_bytes_1 = ds_x_size_1 * ds_y_size_1 * sizeof(int);

    int ds_x_size_2 = x_size>>2;
    int ds_y_size_2 = y_size>>2;
    int ds_bytes_2 = ds_x_size_2 * ds_y_size_2 * sizeof(int);

    // Compute the scattering transform
    scatter(fimage, scheduler,outputFile,
            x_size, y_size, bytes,
            ds_x_size_1, ds_y_size_1, ds_bytes_1,
            ds_x_size_2, ds_y_size_2, ds_bytes_2, separable);
}

int main(int argc, char **argv) {
    // Read file
    char *filename = strdup(DEFAULT_FILENAME);
    if (argc > 1) {
        filename = strdup(argv[1]);
        fprintf(stderr, "Using %s\n", filename);
    }

    // Compute the scattering transform
    JobScheduler scheduler(0);

    //Pre-process for all
    initConsts();
    scheduleForTransformation(&scheduler, filename, "result.ppm");
    scheduleForTransformation(&scheduler, filename, "result.ppm2");
    scheduleForTransformation(&scheduler, filename, "result.ppm3");
    scheduleForTransformation(&scheduler, filename, "result.ppm4");
    scheduleForTransformation(&scheduler, filename, "result.ppm5");

    scheduler.waitUntilDone();
}
