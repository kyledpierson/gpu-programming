#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<random>

#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)

template<typename T>
void _cudaCheck(T e, const char* func, const char* call, const int line){
  if(e != cudaSuccess){
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// Number of elements worked by one thread
long nelem;

// Each threads work on nelem-elements in a pair of sz-long vector
__global__ void add_krnl(float *x, float *y, long sz, long nelem) {
  // PARAMETER: 0 means adjacent, 1 means cyclical
  int pattern = 1;

  long i;
  long inc = (sz + nelem-1) / nelem;
  long start = blockIdx.x*1024 + threadIdx.x;
  
  if (pattern == 0) {
    start = start*nelem;
    long end = start+nelem;
    for(i = start; i < end && i < sz; i++) {
      x[i] += y[i];
    }
  } else {
    for (i = start; i < sz; i += inc) {
      x[i] += y[i];
    }
  }
}

long func_add(float *x, float *y, long sz) {
  long i;

  // CPU Calculation
  for (i = 0; i < sz; i++)
    x[i] += y[i];

  // GPU Calculation
  float *dx, *dy;
  cudaMalloc((float **) &dx, sz*sizeof(float));
  cudaMalloc((float **) &dy, sz*sizeof(float));
  cudaMemcpy(dx, x, sz*sizeof(float), cudaMemcpyHostToDevice);
  
  // Timing using cudaEvent
  cudaEvent_t start, stop;
  float et;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));

  // Time event start
  cudaCheck(cudaEventRecord(start));
  
  {
    long n_threads = (sz + nelem-1) / nelem;
    long n_blocks = (n_threads + 1023) / 1024;

    add_krnl<<<n_blocks, 1024>>>(dx, dy, sz, nelem);
  }

  cudaCheck(cudaGetLastError());

  // Time event end
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&et, start, stop));
  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));

  printf("\t%0.3f", et);

  // Copy data back to d_x and free GPU memory
  float * d_x = (float *) malloc(sz * sizeof(float));
  cudaMemcpy(d_x, dx, sz*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dx);
  cudaFree(dy);

  // Compare CPU and GPU output to see if it is within error tolerance
  for (i = 0; i < sz; i++) {
    if (fabsf(d_x[i] - x[i]) > 1e-5) {
      free(d_x);
      return 0;
    }
  }
  free(d_x);
  return 1;
}


int main(int argc, char **argv) {
  float *a, *b;
  long j;
  long i;

  std::random_device rd;
  std::mt19937_64 mt(rd());
  std::uniform_real_distribution<float> u(0, 1);

  // Print title
  printf("sz");
  for (nelem = 1; nelem < 513; nelem *= 2)
    printf("\t%d", nelem);
  printf("\n");

  for (j = 10; j <= 1000000000; j *= 10) {
    a = (float *) malloc(sizeof(float) * j);
    b = (float *) malloc(sizeof(float) * j);

    // Initialize with random number generator
    for (i = 0; i < j; i++) {
      a[i] = u(mt);
      b[i] = u(mt);
    }

    printf("%d", j);

    for (nelem = 1; nelem < 513; nelem *= 2)
      if (!func_add(a, b, j))
        printf("failed to add\n");

    printf("\n");

    free(a);
    free(b);
  }

  return 0;
}
