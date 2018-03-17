#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"
#include "cuda.h"
#include <complex.h>

#define DEFAULT_THRESHOLD  4000
#define DEFAULT_FILENAME "mountains.ppm"

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
    double gaussian[7][7] = {
        {0.00000019425474,  0.000096568274, 0.00010062644,  0.00021978836,  0.00010062644,  0.000096568274, 0.00000019425474},
        {0.0000096568274,	0.00048006195,	0.0050023603,	0.010926159,	0.0050023603,	0.00048006195,	0.0000096568274},
        {0.00010062644,     0.0050023603,	0.052125789,    0.11385319,	    0.052125789,	0.0050023603,	0.00010062644},
        {0.00021978836,     0.010926159,	0.11385319,	    0.24867822,	    0.11385319,	    0.010926159,    0.00021978836},
        {0.00010062644,     0.0050023603,	0.052125789,    0.11385319,	    0.052125789,	0.0050023603,	0.00010062644},
        {0.0000096568274,	0.00048006195,	0.0050023603,	0.010926159,	0.0050023603,	0.00048006195,	0.0000096568274},
        {0.00000019425474,  0.000096568274, 0.00010062644,  0.00021978836,  0.00010062644,  0.000096568274, 0.00000019425474},
    }

    double complex morlet1[7][7] = {
        {0.0000050278-0.0000066992*I,-0.0000830900+0.0004709791*I,-0.0043360960-0.0034702790*I,0.0088283160+0.0000000000*I,-0.0043360960+0.0034702790*I,-0.0000830900-0.0004709791*I,0.0000050278+0.0000066992*I},
        {0.0000133504-0.0000177885*I,-0.0002206299+0.0012505970*I,-0.0115136900-0.0092146750*I,0.0234419400+0.0000000000*I,-0.0115136900+0.0092146750*I,-0.0002206299-0.0012505970*I,0.0000133504+0.0000177885*I},
        {0.0000239863-0.0000319602*I,-0.0003964000+0.0022469160*I,-0.0206863600-0.0165557700*I,0.0421175400+0.0000000000*I,-0.0206863600+0.0165557700*I,-0.0003964000-0.0022469160*I,0.0000239863+0.0000319602*I},
        {0.0000291599-0.0000388537*I,-0.0004818999+0.0027315550*I,-0.0251482100-0.0201267000*I,0.0512019100+0.0000000000*I,-0.0251482100+0.0201267000*I,-0.0004818999-0.0027315550*I,0.0000291599+0.0000388537*I},
        {0.0000239863-0.0000319602*I,-0.0003964000+0.0022469160*I,-0.0206863600-0.0165557700*I,0.0421175400+0.0000000000*I,-0.0206863600+0.0165557700*I,-0.0003964000-0.0022469160*I,0.0000239863+0.0000319602*I},
        {0.0000133504-0.0000177885*I,-0.0002206299+0.0012505970*I,-0.0115136900-0.0092146750*I,0.0234419400+0.0000000000*I,-0.0115136900+0.0092146750*I,-0.0002206299-0.0012505970*I,0.0000133504+0.0000177885*I},
        {0.0000050278-0.0000066992*I,-0.0000830900+0.0004709791*I,-0.0043360960-0.0034702790*I,0.0088283160+0.0000000000*I,-0.0043360960+0.0034702790*I,-0.0000830900-0.0004709791*I,0.0000050278+0.0000066992*I},
    }

    double complex morlet2[7][7] = {
        {-0.0000002623-0.0000000428*I,0.0000138724-0.0000178732*I,-0.0000014291+0.0007025404*I,-0.0053326850-0.0020938190*I,0.0059336590-0.0043148570*I,-0.0009152358+0.0036726160*I,-0.0003730507-0.0002492096*I},
        {-0.0000020339-0.0000028530*I,0.0001958448+0.0000299994*I,-0.0036912840+0.0032713350*I,-0.0081006680-0.0196504100*I,0.0175430200+0.0084178570*I,-0.0065653990+0.0036563050*I,-0.0000723519-0.0004640453*I},
        {0.0000061847-0.0000236379*I,0.0004391283+0.0010866160*I,-0.0180811300-0.0009752251*I,0.0211026300-0.0368138100*I,0.0042756350+0.0338953400*I,-0.0074747080-0.0020332880*I,0.0001938181-0.0001894189*I},
        {0.0000950236-0.0000291199*I,-0.0020087690+0.0036050660*I,-0.0229593200-0.0254904200*I,0.0515590200-0.0000000000*I,-0.0229593200+0.0254904200*I,-0.0020087690-0.0036050660*I,0.0000950236+0.0000291199*I},
        {0.0001938181+0.0001894189*I,-0.0074747080+0.0020332880*I,0.0042756350-0.0338953400*I,0.0211026300+0.0368138100*I,-0.0180811300+0.0009752251*I,0.0004391283-0.0010866160*I,0.0000061847+0.0000236379*I},
        {-0.0000723519+0.0004640453*I,-0.0065653990-0.0036563050*I,0.0175430200-0.0084178570*I,-0.0081006680+0.0196504100*I,-0.0036912840-0.0032713350*I,0.0001958448-0.0000299994*I,-0.0000020339+0.0000028530*I},
        {-0.0003730507+0.0002492096*I,-0.0009152358-0.0036726160*I,0.0059336590+0.0043148570*I,-0.0053326850+0.0020938190*I,-0.0000014291-0.0007025404*I,0.0000138724+0.0000178732*I,-0.0000002623+0.0000000428*I},
    }

    double complex morlet3[7][7] = {
        {-0.0000000492+0.0000000263*I,-0.0000020442-0.0000028757*I,0.0000613224-0.0000302072*I,0.0000835564+0.0007363237*I,-0.0031549480+0.0005174061*I,-0.0009821461-0.0036447700*I,0.0015283200+0.0000000000*I},
        {-0.0000020442-0.0000028757*I,0.0000906287-0.0000446434*I,0.0001825038+0.0016082790*I,-0.0101843000+0.0016702070*I,-0.0046855570-0.0173882200*I,0.0107757100+0.0000000000*I,-0.0009821461+0.0036447700*I},
        {0.0000613224-0.0000302072*I,0.0001825038+0.0016082790*I,-0.0150514100+0.0024684060*I,-0.0102342000-0.0379793800*I,0.0347844100+0.0000000000*I,-0.0046855570+0.0173882200*I,-0.0031549480-0.0005174061*I},
        {0.0000835564+0.0007363237*I,-0.0101843000+0.0016702070*I,-0.0102342000-0.0379793800*I,0.0514080300-0.0000000000*I,-0.0102342000+0.0379793800*I,-0.0101843000-0.0016702070*I,0.0000835564-0.0007363237*I},
        {-0.0031549480+0.0005174061*I,-0.0046855570-0.0173882200*I,0.0347844100-0.0000000000*I,-0.0102342000+0.0379793800*I,-0.0150514100-0.0024684060*I,0.0001825038-0.0016082790*I,0.0000613224+0.0000302072*I},
        {-0.0009821461-0.0036447700*I,0.0107757100-0.0000000000*I,-0.0046855570+0.0173882200*I,-0.0101843000-0.0016702070*I,0.0001825038-0.0016082790*I,0.0000906287+0.0000446434*I,-0.0000020442+0.0000028757*I},
        {0.0015283200-0.0000000000*I,-0.0009821461+0.0036447700*I,-0.0031549480-0.0005174061*I,0.0000835564-0.0007363237*I,0.0000613224+0.0000302072*I,-0.0000020442+0.0000028757*I,-0.0000000492-0.0000000263*I},
    }

    double complex morlet4[7][7] = {
        {-0.0000002623-0.0000000428*I,-0.0000020339-0.0000028530*I,0.0000061847-0.0000236379*I,0.0000950236-0.0000291199*I,0.0001938181+0.0001894189*I,-0.0000723519+0.0004640453*I,-0.0003730507+0.0002492096*I},
        {0.0000138724-0.0000178732*I,0.0001958448+0.0000299994*I,0.0004391283+0.0010866160*I,-0.0020087690+0.0036050660*I,-0.0074747080+0.0020332880*I,-0.0065653990-0.0036563050*I,-0.0009152358-0.0036726160*I},
        {-0.0000014291+0.0007025404*I,-0.0036912840+0.0032713350*I,-0.0180811300-0.0009752251*I,-0.0229593200-0.0254904200*I,0.0042756350-0.0338953400*I,0.0175430200-0.0084178570*I,0.0059336590+0.0043148570*I},
        {-0.0053326850-0.0020938190*I,-0.0081006680-0.0196504100*I,0.0211026300-0.0368138100*I,0.0515590200+0.0000000000*I,0.0211026300+0.0368138100*I,-0.0081006680+0.0196504100*I,-0.0053326850+0.0020938190*I},
        {0.0059336590-0.0043148570*I,0.0175430200+0.0084178570*I,0.0042756350+0.0338953400*I,-0.0229593200+0.0254904200*I,-0.0180811300+0.0009752251*I,-0.0036912840-0.0032713350*I,-0.0000014291-0.0007025404*I},
        {-0.0009152358+0.0036726160*I,-0.0065653990+0.0036563050*I,-0.0074747080-0.0020332880*I,-0.0020087690-0.0036050660*I,0.0004391283-0.0010866160*I,0.0001958448-0.0000299994*I,0.0000138724+0.0000178732*I},
        {-0.0003730507-0.0002492096*I,-0.0000723519-0.0004640453*I,0.0001938181-0.0001894189*I,0.0000950236+0.0000291199*I,0.0000061847+0.0000236379*I,-0.0000020339+0.0000028530*I,-0.0000002623+0.0000000428*I},
    }

    double complex morlet5[7][7] = {
        {0.0000050278-0.0000066992*I,0.0000133504-0.0000177885*I,0.0000239863-0.0000319602*I,0.0000291599-0.0000388537*I,0.0000239863-0.0000319602*I,0.0000133504-0.0000177885*I,0.0000050278-0.0000066992*I},
        {-0.0000830900+0.0004709791*I,-0.0002206299+0.0012505970*I,-0.0003964000+0.0022469160*I,-0.0004818999+0.0027315550*I,-0.0003964000+0.0022469160*I,-0.0002206299+0.0012505970*I,-0.0000830900+0.0004709791*I},
        {-0.0043360960-0.0034702790*I,-0.0115136900-0.0092146750*I,-0.0206863600-0.0165557700*I,-0.0251482100-0.0201267000*I,-0.0206863600-0.0165557700*I,-0.0115136900-0.0092146750*I,-0.0043360960-0.0034702790*I},
        {0.0088283160-0.0000000000*I,0.0234419400-0.0000000000*I,0.0421175400-0.0000000000*I,0.0512019100-0.0000000000*I,0.0421175400+0.0000000000*I,0.0234419400+0.0000000000*I,0.0088283160+0.0000000000*I},
        {-0.0043360960+0.0034702790*I,-0.0115136900+0.0092146750*I,-0.0206863600+0.0165557700*I,-0.0251482100+0.0201267000*I,-0.0206863600+0.0165557700*I,-0.0115136900+0.0092146750*I,-0.0043360960+0.0034702790*I},
        {-0.0000830900-0.0004709791*I,-0.0002206299-0.0012505970*I,-0.0003964000-0.0022469160*I,-0.0004818999-0.0027315550*I,-0.0003964000-0.0022469160*I,-0.0002206299-0.0012505970*I,-0.0000830900-0.0004709791*I},
        {0.0000050278+0.0000066992*I,0.0000133504+0.0000177885*I,0.0000239863+0.0000319602*I,0.0000291599+0.0000388537*I,0.0000239863+0.0000319602*I,0.0000133504+0.0000177885*I,0.0000050278+0.0000066992*I},
    }

    double complex morlet6[7][7] = {
        {-0.0003730507+0.0002492096*I,-0.0000723519+0.0004640453*I,0.0001938181+0.0001894189*I,0.0000950236-0.0000291199*I,0.0000061847-0.0000236379*I,-0.0000020339-0.0000028530*I,-0.0000002623-0.0000000428*I},
        {-0.0009152358-0.0036726160*I,-0.0065653990-0.0036563050*I,-0.0074747080+0.0020332880*I,-0.0020087690+0.0036050660*I,0.0004391283+0.0010866160*I,0.0001958448+0.0000299994*I,0.0000138724-0.0000178732*I},
        {0.0059336590+0.0043148570*I,0.0175430200-0.0084178570*I,0.0042756350-0.0338953400*I,-0.0229593200-0.0254904200*I,-0.0180811300-0.0009752251*I,-0.0036912840+0.0032713350*I,-0.0000014291+0.0007025404*I},
        {-0.0053326850+0.0020938190*I,-0.0081006680+0.0196504100*I,0.0211026300+0.0368138100*I,0.0515590200+0.0000000000*I,0.0211026300-0.0368138100*I,-0.0081006680-0.0196504100*I,-0.0053326850-0.0020938190*I},
        {-0.0000014291-0.0007025404*I,-0.0036912840-0.0032713350*I,-0.0180811300+0.0009752251*I,-0.0229593200+0.0254904200*I,0.0042756350+0.0338953400*I,0.0175430200+0.0084178570*I,0.0059336590-0.0043148570*I},
        {0.0000138724+0.0000178732*I,0.0001958448-0.0000299994*I,0.0004391283-0.0010866160*I,-0.0020087690-0.0036050660*I,-0.0074747080-0.0020332880*I,-0.0065653990+0.0036563050*I,-0.0009152358+0.0036726160*I},
        {-0.0000002623+0.0000000428*I,-0.0000020339+0.0000028530*I,0.0000061847+0.0000236379*I,0.0000950236+0.0000291199*I,0.0001938181-0.0001894189*I,-0.0000723519-0.0004640453*I,-0.0003730507-0.0002492096*I},
    }

    double complex morlet7[7][7] = {
        {0.0015283200-0.0000000000*I,-0.0009821461-0.0036447700*I,-0.0031549480+0.0005174061*I,0.0000835564+0.0007363237*I,0.0000613224-0.0000302072*I,-0.0000020442-0.0000028757*I,-0.0000000492+0.0000000263*I},
        {-0.0009821461+0.0036447700*I,0.0107757100-0.0000000000*I,-0.0046855570-0.0173882200*I,-0.0101843000+0.0016702070*I,0.0001825038+0.0016082790*I,0.0000906287-0.0000446434*I,-0.0000020442-0.0000028757*I},
        {-0.0031549480-0.0005174061*I,-0.0046855570+0.0173882200*I,0.0347844100-0.0000000000*I,-0.0102342000-0.0379793800*I,-0.0150514100+0.0024684060*I,0.0001825038+0.0016082790*I,0.0000613224-0.0000302072*I},
        {0.0000835564-0.0007363237*I,-0.0101843000-0.0016702070*I,-0.0102342000+0.0379793800*I,0.0514080300-0.0000000000*I,-0.0102342000-0.0379793800*I,-0.0101843000+0.0016702070*I,0.0000835564+0.0007363237*I},
        {0.0000613224+0.0000302072*I,0.0001825038-0.0016082790*I,-0.0150514100-0.0024684060*I,-0.0102342000+0.0379793800*I,0.0347844100+0.0000000000*I,-0.0046855570-0.0173882200*I,-0.0031549480+0.0005174061*I},
        {-0.0000020442+0.0000028757*I,0.0000906287+0.0000446434*I,0.0001825038-0.0016082790*I,-0.0101843000-0.0016702070*I,-0.0046855570+0.0173882200*I,0.0107757100+0.0000000000*I,-0.0009821461-0.0036447700*I},
        {-0.0000000492-0.0000000263*I,-0.0000020442+0.0000028757*I,0.0000613224+0.0000302072*I,0.0000835564-0.0007363237*I,-0.0031549480-0.0005174061*I,-0.0009821461+0.0036447700*I,0.0015283200+0.0000000000*I},
    }

    double complex morlet8[7][7] = {
        {-0.0003730507-0.0002492096*I,-0.0009152358+0.0036726160*I,0.0059336590-0.0043148570*I,-0.0053326850-0.0020938190*I,-0.0000014291+0.0007025404*I,0.0000138724-0.0000178732*I,-0.0000002623-0.0000000428*I},
        {-0.0000723519-0.0004640453*I,-0.0065653990+0.0036563050*I,0.0175430200+0.0084178570*I,-0.0081006680-0.0196504100*I,-0.0036912840+0.0032713350*I,0.0001958448+0.0000299994*I,-0.0000020339-0.0000028530*I},
        {0.0001938181-0.0001894189*I,-0.0074747080-0.0020332880*I,0.0042756350+0.0338953400*I,0.0211026300-0.0368138100*I,-0.0180811300-0.0009752251*I,0.0004391283+0.0010866160*I,0.0000061847-0.0000236379*I},
        {0.0000950236+0.0000291199*I,-0.0020087690-0.0036050660*I,-0.0229593200+0.0254904200*I,0.0515590200-0.0000000000*I,-0.0229593200-0.0254904200*I,-0.0020087690+0.0036050660*I,0.0000950236-0.0000291199*I},
        {0.0000061847+0.0000236379*I,0.0004391283-0.0010866160*I,-0.0180811300+0.0009752251*I,0.0211026300+0.0368138100*I,0.0042756350-0.0338953400*I,-0.0074747080+0.0020332880*I,0.0001938181+0.0001894189*I},
        {-0.0000020339+0.0000028530*I,0.0001958448-0.0000299994*I,-0.0036912840-0.0032713350*I,-0.0081006680+0.0196504100*I,0.0175430200-0.0084178570*I,-0.0065653990-0.0036563050*I,-0.0000723519+0.0004640453*I},
        {-0.0000002623+0.0000000428*I,0.0000138724+0.0000178732*I,-0.0000014291-0.0007025404*I,-0.0053326850+0.0020938190*I,0.0059336590+0.0043148570*I,-0.0009152358-0.0036726160*I,-0.0003730507+0.0002492096*I},
    }

    // ===================== READ PARAMETERS ======================
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

    // Read image
    int xsize, ysize, maxval;
    unsigned int *pic = read_ppm( filename, xsize, ysize, maxval );
    int numbytes =  xsize * ysize * 3 * sizeof( int );

    // ==================== GPU IMPLEMENTATION ====================
    char* strategy = "unrolled";

    int yround = ysize;
    int xround = xsize;
    int ythreads = 32;
    int xthreads = 32;
    int yelems = 4;
    int xelems = 4;
    int ymult = ythreads*yelems;
    int xmult = xthreads*xelems;

    // For simplicity, have exactly the number of blocks and threads needed
    if (yround % ymult) {
        yround = yround/ymult*ymult + ymult;
    }
    if (xround % xmult) {
        xround = xround/xmult*xmult + xmult;
    }
    int yblocks = yround / ymult;
    int xblocks = xround / xmult;

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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    sobel_unrolled<<<blocks, threads>>>(dPic, dResult, thresh, ysize, xsize, ythreads, xthreads, yelems, xelems);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time,start, stop);

    // Copy the result
    cudaMemcpy(resultGPU, dResult, numbytes, cudaMemcpyDeviceToHost);

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
    free(resultGPU);
    free(name);
}

