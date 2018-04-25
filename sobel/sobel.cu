#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"
#include "cuda.h"

#define DEFAULT_THRESHOLD  4000
#define DEFAULT_FILENAME "images/BWstop-sign.ppm"

unsigned int *read_ppm( char *filename, int & xsize, int & ysize, int & maxval ) {
    if ( !filename || filename[0] == '\0') {
        fprintf(stderr, "read_ppm but no file name\n");
        return NULL;  // fail
    }

    fprintf(stderr, "read_ppm( %s )\n", filename);
    int fd = open( filename, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
        return NULL; // fail
    }

    char chars[1024];
    int num = read(fd, chars, 1000);

    if (chars[0] != 'P' || chars[1] != '6') {
        fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
        return NULL;
    }

    unsigned int width, height, maxvalue;
    char *ptr = chars+3; // P 6 newline
    if (*ptr == '#') { // comment line!
        ptr = 1 + strstr(ptr, "\n");
    }

    num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
    fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);
    xsize = width;
    ysize = height;
    maxval = maxvalue;

    unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
    if (!pic) {
        fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
        return NULL; // fail but return
    }

    // allocate buffer to read the rest of the file into
    int bufsize =  3 * width * height * sizeof(unsigned char);
    if (maxval > 255) bufsize *= 2;
    unsigned char *buf = (unsigned char *)malloc( bufsize );
    if (!buf) {
        fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
        return NULL; // fail but return
    }

    // TODO really read
    char duh[80];
    char *line = chars;

    // find the start of the pixel data.   no doubt stupid
    sprintf(duh, "%d\0", xsize);
    line = strstr(line, duh);
    //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
    line += strlen(duh) + 1;

    sprintf(duh, "%d\0", ysize);
    line = strstr(line, duh);
    //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
    line += strlen(duh) + 1;

    sprintf(duh, "%d\0", maxval);
    line = strstr(line, duh);

    fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
    line += strlen(duh) + 1;

    long offset = line - chars;
    lseek(fd, offset, SEEK_SET); // move to the correct offset
    long numread = read(fd, buf, bufsize);
    fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize);

    close(fd);

    int pixels = xsize * ysize;
    for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel

    return pic; // success
}

void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) {
    FILE *fp;

    fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
        exit(-1);
    }
    // int x,y;

    fprintf(fp, "P6\n");
    fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);

    int numpix = xsize * ysize;
    for (int i=0; i<numpix; i++) {
        unsigned char uc = (unsigned char) pic[i];
        fprintf(fp, "%c%c%c", uc, uc, uc);
    }
    fclose(fp);
}

// ======================================== SOBEL ========================================
__device__ void sobel(int offset, int thresh, int *result,
                      unsigned int i0j0, unsigned int i0j1, unsigned int i0j2,
                      unsigned int i1j0, unsigned int i1j1, unsigned int i1j2,
                      unsigned int i2j0, unsigned int i2j1, unsigned int i2j2) {
    int sum1 = i0j2 -   i0j0 + 2*i1j2 - 2*i1j0 +   i2j2 - i2j0;
    int sum2 = i0j0 + 2*i0j1 +   i0j2 -   i2j0 - 2*i2j1 - i2j2;
    int magnitude =  sum1*sum1 + sum2*sum2;

    if (magnitude > thresh) {
        result[offset] = 255;
    }
}

/*
Naive solution
Converts the outer loop of the sequential code into blocks, and the inner loop into threads
*/
__global__ void sobel_naive(unsigned int *pic, int *result, int thresh,
                            int ysize, int xsize, int ythreads, int xthreads, int yelems, int xelems) {
    int i, j, offset, sum1, sum2, magnitude;

    for (i = blockIdx.x + 1; i < ysize - 1; i += gridDim.x) {
        for (j = threadIdx.x + 1; j < xsize - 1; j += xthreads) {
            offset = i * xsize + j;

            sum1 = pic[xsize*(i-1) + j+1] -   pic[xsize*(i-1) + j-1]
               + 2*pic[xsize* i    + j+1] - 2*pic[xsize* i    + j-1]
               +   pic[xsize*(i+1) + j+1] -   pic[xsize*(i+1) + j-1];

            sum2 = pic[xsize*(i-1) + j-1] + 2*pic[xsize*(i-1) + j] + pic[xsize*(i-1) + j+1]
                 - pic[xsize*(i+1) + j-1] - 2*pic[xsize*(i+1) + j] - pic[xsize*(i+1) + j+1];

            magnitude =  sum1*sum1 + sum2*sum2;
            if (magnitude > thresh) {
                result[offset] = 255;
            }
        }
    }
}

/*
Shared solution
Each thread in a block brings one element into shared memory, with the perimeter threads bringing
additional halo elements (divergent threads).  Then each thread computes sobel for its element.
*/
__global__ void sobel_shared(unsigned int *pic, int *result, int thresh,
                             int ysize, int xsize, int ythreads, int xthreads, int yelems, int xelems) {
    __shared__ unsigned int tile[1156];

    int y = threadIdx.y;
    int x = threadIdx.x;
    int rowsize = xthreads+2;

    int min_y = blockIdx.y*ythreads+1;
    int min_x = blockIdx.x*xthreads+1;
    int max_y = min(ysize-1, min_y+ythreads-1);
    int max_x = min(xsize-1, min_x+xthreads-1);

    // Load into shared memory
    if (y == 0) {
        // Top-left corner
        if (x == 0) {
            tile[0] = pic[xsize*(min_y-1) + (min_x-1)];
        }
        // Bring in first row
        tile[x+1] = pic[xsize*(min_y-1) + (min_x+x)];
    } else if (y == ythreads-1) {
        // Bottom-right corner
        if (x == xthreads-1) {
            tile[rowsize*(y+2) + x+2] = pic[xsize*(max_y+1) + (max_x+1)];
        }
        // Bring in last row
        tile[rowsize*(y+2) + x+1] = pic[xsize*(max_y+1) + (min_x+x)];
    }
    if (x == 0) {
        // Bottom-left corner
        if (y == ythreads-1) {
            tile[rowsize*(y+2)] = pic[xsize*(max_y+1) + (min_x-1)];
        }
        // Bring in first column
        tile[rowsize*(y+1)] = pic[xsize*(min_y+y) + (min_x-1)];
    } else if (x == xthreads-1) {
        // Top-right corner
        if (y == 0) {
            tile[x+2] = pic[xsize*(min_y-1) + (max_x+1)];
        }
        // Bring in last column
        tile[rowsize*(y+1) + x+2] = pic[xsize*(min_y+y) + (max_x+1)];
    }
    tile[rowsize*(y+1) + (x+1)] = pic[xsize*(min_y+y) + (min_x+x)];

    // Sync the threads
    __syncthreads();

    // Each thread computes sobel
    int sum1 = tile[rowsize* y    + (x+2)] -   tile[rowsize* y    + x]
           + 2*tile[rowsize*(y+1) + (x+2)] - 2*tile[rowsize*(y+1) + x]
           +   tile[rowsize*(y+2) + (x+2)] -   tile[rowsize*(y+2) + x];

    int sum2 = tile[rowsize* y    + x] + 2*tile[rowsize* y    + (x+1)] + tile[rowsize* y    + (x+2)]
             - tile[rowsize*(y+2) + x] - 2*tile[rowsize*(y+2) + (x+1)] - tile[rowsize*(y+2) + (x+2)];

    int magnitude =  sum1*sum1 + sum2*sum2;
    if (magnitude > thresh) {
        result[xsize*(min_y+y) + (min_x+x)] = 255;
    }
}

/*
Shared overlapping solution
Each thread in a block brings one element into shared memory.  Then the interior threads compute
sobel for each of their elements.  Blocks must overlap to compute sobel for all elements.
*/
__global__ void sobel_shared_overlap(unsigned int *pic, int *result, int thresh,
                                     int ysize, int xsize, int ythreads, int xthreads, int yelems, int xelems) {
    __shared__ unsigned int tile[1024];

    int y = threadIdx.y;
    int x = threadIdx.x;

    int min_y = blockIdx.y*(ythreads-2);
    int min_x = blockIdx.x*(xthreads-2);

    // Load into shared memory
    tile[xthreads*y + x] = pic[xsize*(min_y+y) + (min_x+x)];

    // Sync the threads
    __syncthreads();

    // Each interior thread computes sobel
    if (y > 0 && y < ythreads-1 && x > 0 && x < xthreads-1) {
        int sum1 = tile[xthreads*(y-1) + (x+1)] -   tile[xthreads*(y-1) + (x-1)]
               + 2*tile[xthreads* y    + (x+1)] - 2*tile[xthreads* y    + (x-1)]
               +   tile[xthreads*(y+1) + (x+1)] -   tile[xthreads*(y+1) + (x-1)];

        int sum2 = tile[xthreads*(y-1) + (x-1)] + 2*tile[xthreads*(y-1) + x] + tile[xthreads*(y-1) + (x+1)]
                 - tile[xthreads*(y+1) + (x-1)] - 2*tile[xthreads*(y+1) + x] - tile[xthreads*(y+1) + (x+1)];

        int magnitude =  sum1*sum1 + sum2*sum2;
        if (magnitude > thresh) {
            result[xsize*(min_y+y) + (min_x+x)] = 255;
        }
    }
}

/*
Unrolled solution
Each thread loads a 6x6 grid into registers, then computes sobel for each of the 4x4 interior elements
*/
__global__ void sobel_unrolled(unsigned int *pic, int *result, int thresh,
                               int ysize, int xsize, int ythreads, int xthreads, int yelems, int xelems) {
    // Get bounds
    int min_y = blockIdx.y * ythreads * yelems + 1;
    int min_x = blockIdx.x * xthreads * xelems + 1;

    // Put elements into registers
    int i = min_y + threadIdx.y * yelems;
    int j = min_x + threadIdx.x * xelems;

    unsigned int i0j0 = pic[xsize*(i-1) + (j-1)];
    unsigned int i0j1 = pic[xsize*(i-1) +  j];
    unsigned int i0j2 = pic[xsize*(i-1) + (j+1)];
    unsigned int i0j3 = pic[xsize*(i-1) + (j+2)];
    unsigned int i0j4 = pic[xsize*(i-1) + (j+3)];
    unsigned int i0j5 = pic[xsize*(i-1) + (j+4)];

    unsigned int i1j0 = pic[xsize* i    + (j-1)];
    unsigned int i1j1 = pic[xsize* i    +  j];
    unsigned int i1j2 = pic[xsize* i    + (j+1)];
    unsigned int i1j3 = pic[xsize* i    + (j+2)];
    unsigned int i1j4 = pic[xsize* i    + (j+3)];
    unsigned int i1j5 = pic[xsize* i    + (j+4)];

    unsigned int i2j0 = pic[xsize*(i+1) + (j-1)];
    unsigned int i2j1 = pic[xsize*(i+1) +  j];
    unsigned int i2j2 = pic[xsize*(i+1) + (j+1)];
    unsigned int i2j3 = pic[xsize*(i+1) + (j+2)];
    unsigned int i2j4 = pic[xsize*(i+1) + (j+3)];
    unsigned int i2j5 = pic[xsize*(i+1) + (j+4)];

    unsigned int i3j0 = pic[xsize*(i+2) + (j-1)];
    unsigned int i3j1 = pic[xsize*(i+2) +  j];
    unsigned int i3j2 = pic[xsize*(i+2) + (j+1)];
    unsigned int i3j3 = pic[xsize*(i+2) + (j+2)];
    unsigned int i3j4 = pic[xsize*(i+2) + (j+3)];
    unsigned int i3j5 = pic[xsize*(i+2) + (j+4)];

    unsigned int i4j0 = pic[xsize*(i+3) + (j-1)];
    unsigned int i4j1 = pic[xsize*(i+3) +  j];
    unsigned int i4j2 = pic[xsize*(i+3) + (j+1)];
    unsigned int i4j3 = pic[xsize*(i+3) + (j+2)];
    unsigned int i4j4 = pic[xsize*(i+3) + (j+3)];
    unsigned int i4j5 = pic[xsize*(i+3) + (j+4)];

    unsigned int i5j0 = pic[xsize*(i+4) + (j-1)];
    unsigned int i5j1 = pic[xsize*(i+4) +  j];
    unsigned int i5j2 = pic[xsize*(i+4) + (j+1)];
    unsigned int i5j3 = pic[xsize*(i+4) + (j+2)];
    unsigned int i5j4 = pic[xsize*(i+4) + (j+3)];
    unsigned int i5j5 = pic[xsize*(i+4) + (j+4)];

    sobel(i * xsize + j, thresh, result,
          i0j0, i0j1, i0j2, i1j0, i1j1, i1j2, i2j0, i2j1, i2j2);
    sobel(i * xsize + (j+1), thresh, result,
          i0j1, i0j2, i0j3, i1j1, i1j2, i1j3, i2j1, i2j2, i2j3);
    sobel(i * xsize + (j+2), thresh, result,
          i0j2, i0j3, i0j4, i1j2, i1j3, i1j4, i2j2, i2j3, i2j4);
    sobel(i * xsize + (j+3), thresh, result,
          i0j3, i0j4, i0j5, i1j3, i1j4, i1j5, i2j3, i2j4, i2j5);

    sobel((i+1) * xsize + j, thresh, result,
          i1j0, i1j1, i1j2, i2j0, i2j1, i2j2, i3j0, i3j1, i3j2);
    sobel((i+1) * xsize + (j+1), thresh, result,
          i1j1, i1j2, i1j3, i2j1, i2j2, i2j3, i3j1, i3j2, i3j3);
    sobel((i+1) * xsize + (j+2), thresh, result,
          i1j2, i1j3, i1j4, i2j2, i2j3, i2j4, i3j2, i3j3, i3j4);
    sobel((i+1) * xsize + (j+3), thresh, result,
          i1j3, i1j4, i1j5, i2j3, i2j4, i2j5, i3j3, i3j4, i3j5);

    sobel((i+2) * xsize + j, thresh, result,
          i2j0, i2j1, i2j2, i3j0, i3j1, i3j2, i4j0, i4j1, i4j2);
    sobel((i+2) * xsize + (j+1), thresh, result,
          i2j1, i2j2, i2j3, i3j1, i3j2, i3j3, i4j1, i4j2, i4j3);
    sobel((i+2) * xsize + (j+2), thresh, result,
          i2j2, i2j3, i2j4, i3j2, i3j3, i3j4, i4j2, i4j3, i4j4);
    sobel((i+2) * xsize + (j+3), thresh, result,
          i2j3, i2j4, i2j5, i3j3, i3j4, i3j5, i4j3, i4j4, i4j5);

    sobel((i+3) * xsize + j, thresh, result,
          i3j0, i3j1, i3j2, i4j0, i4j1, i4j2, i5j0, i5j1, i5j2);
    sobel((i+3) * xsize + (j+1), thresh, result,
          i3j1, i3j2, i3j3, i4j1, i4j2, i4j3, i5j1, i5j2, i5j3);
    sobel((i+3) * xsize + (j+2), thresh, result,
          i3j2, i3j3, i3j4, i4j2, i4j3, i4j4, i5j2, i5j3, i5j4);
    sobel((i+3) * xsize + (j+3), thresh, result,
          i3j3, i3j4, i3j5, i4j3, i4j4, i4j5, i5j3, i5j4, i5j5);
}

int main( int argc, char **argv ) {
    int thresh = DEFAULT_THRESHOLD;
    char *filename;
    filename = strdup( DEFAULT_FILENAME);

    if (argc > 1) {
        if (argc == 3)  { // filename AND threshold
            filename = strdup( argv[1]);
            thresh = atoi( argv[2] );
        }
        if (argc == 2) { // default file but specified threshhold
            thresh = atoi( argv[1] );
        }
        fprintf(stderr, "file %s    threshold %d\n", filename, thresh);
    }

    int xsize, ysize, maxval;
    unsigned int *pic = read_ppm( filename, xsize, ysize, maxval );

    int numbytes =  xsize * ysize * 3 * sizeof( int );
    int *result = (int *) malloc( numbytes );
    if (!result) {
        fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
        exit(-1); // fail
    }

    int i, j, magnitude, sum1, sum2;
    int *out = result;

    for (int col=0; col<ysize; col++) {
        for (int row=0; row<xsize; row++) {
            *out++ = 0;
        }
    }

    for (i = 1;  i < ysize - 1; i++) {
        for (j = 1; j < xsize -1; j++) {
            int offset = i*xsize + j;

            sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ]
              + 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
              +     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];

            sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
                 - pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];

            magnitude =  sum1*sum1 + sum2*sum2;

            if (magnitude > thresh)
                result[offset] = 255;
            else
                result[offset] = 0;
        }
    }

    write_ppm( "result.ppm", xsize, ysize, 255, result);
    fprintf(stderr, "sobel done\n");

    // ==================== GPU IMPLEMENTATION ====================
    char* strategy = "unrolled";

    int yround = ysize;
    int xround = xsize;
    int yblocks, xblocks, ymult, xmult, ythreads, xthreads, yelems, xelems;

    if (strategy == "naive") {
        yblocks = 1;
        ythreads = 1;
        xblocks = ysize;
        xthreads = xsize;
        if (xblocks > 1024) {
            xblocks = 1024;
        }
        if (xthreads > 1024) {
            xthreads = 1024;
        }
    } else {
        ythreads = 32;
        xthreads = 32;
        if (strategy == "shared") {
            ymult = ythreads;
            xmult = xthreads;
        } else if (strategy == "shared_overlap") {
            ymult = ythreads-2;
            xmult = xthreads-2;
        } else if (strategy == "unrolled") {
            yelems = 4;
            xelems = 4;
            ymult = ythreads*yelems;
            xmult = xthreads*xelems;
        }
        // For simplicity, have exactly the number of blocks and threads needed
        if (yround % ymult) {
            yround = yround/ymult*ymult + ymult;
        }
        if (xround % xmult) {
            xround = xround/xmult*xmult + xmult;
        }
        yblocks = yround / ymult;
        xblocks = xround / xmult;
    }

    dim3 blocks(xblocks, yblocks);
    dim3 threads(xthreads, ythreads);

    // Allocate memory
    unsigned int *dPic;
    int *dResult;
    int *resultGPU = (int *) malloc(numbytes);
    if (!resultGPU) {
        fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
        exit(-1); // fail
    }

    cudaMalloc((unsigned int **) &dPic, yround*xround*sizeof(unsigned int));
    cudaMalloc((int **) &dResult, numbytes);

    cudaMemcpy(dPic, pic, ysize*xsize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(dResult, 0, numbytes);

    float elapsed_time;
    cudaEvent_t start,stop;

    // Execute the kernel
    if (strategy == "naive") {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);

        sobel_naive<<<blocks, threads>>>(dPic, dResult, thresh, ysize, xsize, ythreads, xthreads, yelems, xelems);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time,start, stop);
    } else if (strategy == "shared") {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);

        sobel_shared<<<blocks, threads>>>(dPic, dResult, thresh, ysize, xsize, ythreads, xthreads, yelems, xelems);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time,start, stop);
    } else if (strategy == "shared_overlap") {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);

        sobel_shared_overlap<<<blocks, threads>>>(dPic, dResult, thresh, ysize, xsize, ythreads, xthreads, yelems, xelems);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time,start, stop);
    } else if (strategy == "unrolled") {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);

        sobel_unrolled<<<blocks, threads>>>(dPic, dResult, thresh, ysize, xsize, ythreads, xthreads, yelems, xelems);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time,start, stop);
    }

    // Copy the result
    cudaMemcpy(resultGPU, dResult, numbytes, cudaMemcpyDeviceToHost);

    for (i = 1; i < ysize - 1; i++) {
        for (j = 1; j < xsize - 1; j++) {
            if (result[i*xsize + j] != resultGPU[i*xsize + j]) {
                //fprintf(stderr, "RESULTS NOT EQUAL %i %i\n", i, j);
            }
        }
    }

    // Write the ppm file
    char* name = (char*) malloc(strlen(strategy) + 5);
    if (!name) {
        fprintf(stderr, "sobel() unable to malloc %d bytes\n", strlen(strategy) + 5);
        exit(-1); // fail
    }

    strcpy(name, strategy);
    strcat(name, ".ppm");
    write_ppm(name, xsize, ysize, 255, resultGPU);
    fprintf(stderr, "%4.4f\n", elapsed_time);

    // Free memory
    cudaFree(dPic);
    cudaFree(dResult);
    free(pic);
    free(result);
    free(resultGPU);
    free(name);
}

