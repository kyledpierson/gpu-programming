#ifndef IOHANDLER_H
#define IOHANDLER_H

void* memCheck(void* mem);
unsigned int *read_ppm(char *filename, int & xsize, int & ysize, int & maxval);
void write_ppm(char *filename, int xsize, int ysize, int maxval, int *pic);

#endif
