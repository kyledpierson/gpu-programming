#ifndef IOHANDLER_H
#define IOHANDLER_H

void* mem_check(void* mem);
void read_filter(char *filename, float* filter);
unsigned int *read_ppm(char *filename, int & xsize, int & ysize, int & maxval);
void write_ppm(char *filename, int xsize, int ysize, int maxval, int *pic);

#endif
