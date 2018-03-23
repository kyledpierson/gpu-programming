#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include "string.h"
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
    int xsize, ysize, maxval;
    unsigned int *image = read_ppm(filename, xsize, ysize, maxval);
    int bytes = ysize*xsize*sizeof(int);

    // Account for downsampling
    int dsysize = ysize>>1;
    int dsxsize = xsize>>1;
    int dsbytes = dsysize*dsxsize*sizeof(int);
    int *result = (int*) memCheck(malloc(dsbytes));

    // Compute the scattering transform
    scatter(image, result, xsize, ysize, bytes, dsxsize, dsysize, dsbytes);

    // Write the result
    write_ppm("result.ppm", dsxsize, dsysize, 255, result);

    // Free memory
    free(image);
    free(result);
}
