#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <unistd.h>

#include "iohandler.h"
#include "scatter.h"
#include "JobScheduler.h"

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

    // Account for downsampling
    int ds_x_size = x_size>>1;
    int ds_y_size = y_size>>1;

    int ds_bytes = ds_x_size * ds_y_size * sizeof(int);
    int *result = (int*) mem_check(malloc(ds_bytes));

    // Compute the scattering transform
    // MATLAB SEQUENTIAL CODE TIME: 1.0044830 seconds
    // 2D SCATTER TIME:             0.0033391 seconds
    // SEPARABLE SCATTER TIME:      0.0008100 seconds

    //2 gigs, 1 job max
    JobScheduler scheduler(2 * 1024 * 1024 * 1024,1);

    scatter(&scheduler, image, result, x_size, y_size, bytes, ds_x_size, ds_y_size, ds_bytes);
    // scatter_separable(image, result, x_size, y_size, bytes, ds_x_size, ds_y_size, ds_bytes);

    // Write the result
    write_ppm("result.ppm", ds_x_size, ds_y_size, 255, result);

    // Free memory
    free(image);
    free(result);
}
