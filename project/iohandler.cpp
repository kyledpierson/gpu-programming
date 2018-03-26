#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <unistd.h>

#include "iohandler.h"

void* mem_check(void* mem) {
    // Make sure memory is allocated
    if (!mem) {
        fprintf(stderr, "Unable to allocate bytes\n");
        exit(-1);
    }
    return mem;
}

unsigned int *read_ppm(char *filename, int & xsize, int & ysize, int & maxval) {
    if (!filename || filename[0] == '\0') {
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

    unsigned int *pic = (unsigned int *) mem_check(malloc( width * height * sizeof(unsigned int)));

    // allocate buffer to read the rest of the file into
    int bufsize =  3 * width * height * sizeof(unsigned char);
    if (maxval > 255) bufsize *= 2;
    unsigned char *buf = (unsigned char *) mem_check(malloc(bufsize));

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

void write_ppm(char *filename, int xsize, int ysize, int maxval, int *pic) {
    FILE *fp;

    fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
        exit(-1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);

    int numpix = xsize * ysize;
    for (int i=0; i<numpix; i++) {
        unsigned char uc = (unsigned char) pic[i];
        fprintf(fp, "%c%c%c", uc, uc, uc);
    }
    fclose(fp);
}
