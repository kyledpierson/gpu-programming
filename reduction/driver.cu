// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
#include<stdio.h>
#include "cuda.h"
#include<string.h>
#include<stdlib.h>

#define BLOCK_SIZE 512

char *inputFile,*outputFile;
void _errorCheck(cudaError_t e) {
    if(e != cudaSuccess) {
        printf("Failed to run statement \n");
    }
}

// ==================================================================
__global__ void total_sequential(float *input, float *output, int len) {
    if (threadIdx.x == 0) {
        int i;
        int sum = 0;
        int block = blockIdx.x;

        int elems = (len + gridDim.x - 1) / gridDim.x;
        int start = block * elems;
        int stop = min(start + elems, len);

        // One thread computes sum
        for (i = start; i < stop; i++) {
            sum += input[i];
        }
        output[block] = sum;
    }
}

__global__ void total_sequential_coalesced(float *input, float *output, int len) {
    if (threadIdx.x == 0) {
        int i;
        int sum = 0;

        int start = blockIdx.x;
        int step = gridDim.x;

        // One thread computes sum
        for (i = start; i < len; i += step) {
            sum += input[i];
        }
        output[start] = sum;
    }
}

__global__ void total_atomic(float *input, float *output, int len) {
    int i;
    int block = blockIdx.x;
    int thread = threadIdx.x;

    int elems = (len + gridDim.x - 1) / gridDim.x;
    int start = block * elems;
    int stop = min(start + elems, len);

    // Thread computes adjacent elements
    int elems_per_thread = (elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    start = start + thread * elems_per_thread;
    stop = min(stop, start + elems_per_thread);

    for (i = start; i < stop; i++) {
        // Every thread atomically updates the sum
        atomicAdd(output + block, input[i]);
    }
}

__global__ void total_atomic_coalesced(float *input, float *output, int len) {
    int i;
    int block = blockIdx.x;
    int thread = threadIdx.x;

    int start = block * BLOCK_SIZE + thread;
    int step = gridDim.x * BLOCK_SIZE;

    for (i = start; i < len; i+= step) {
        // Every thread atomically updates the sum
        atomicAdd(output + block, input[i]);
    }
}

__global__ void total_partial_reduction(float *input, float *output, int len) {
    __shared__ float shared[BLOCK_SIZE];

    int i;
    int block = blockIdx.x;
    int thread = threadIdx.x;

    int start = block * BLOCK_SIZE + thread;
    int step = gridDim.x * BLOCK_SIZE;

    // Each thread computes a sum in shared memory
    shared[thread] = 0;
    for (i = start; i < len; i += step) {
        shared[thread] += input[i];
    }
    __syncthreads();

    // Compute sum for blocks on thread 0
    if (thread == 0) {
        int sum = 0;
        for (i = 0; i < BLOCK_SIZE; i++) {
            sum += shared[i];
        }
        output[block] = sum;
    }
}

__global__ void total_reduction(float *input, float *output, int len) {
    __shared__ float shared[BLOCK_SIZE];

    int i;
    int block = blockIdx.x;
    int thread = threadIdx.x;

    int start = block * 2 * BLOCK_SIZE + thread;
    int step = gridDim.x * 2 * BLOCK_SIZE;

    // Many elements per thread
    shared[thread] = 0;
    for (i = start; i < len; i += step) {
        shared[thread] += input[i] + input[i + BLOCK_SIZE];
    }
    __syncthreads();

    // Recursively reduce down to 1 element
    for (i = BLOCK_SIZE >> 1; i > 0; i >>= 1) {
        if (thread < i) {
            shared[thread] += shared[thread + i];
        }
        __syncthreads();
    }

    if (thread == 0) {
        output[block] = shared[0];
    }
}
// ==================================================================

void parseInput(int argc, char **argv) {
    if(argc < 2) {
        printf("Not enough arguments\n");
        printf("Usage: reduction -i inputFile -o outputFile\n");
        exit(1);
    }
    int i=1;
    while(i<argc) {
        if(!strcmp(argv[i],"-i")) {
            ++i;
            inputFile = argv[i];
        } else if(!strcmp(argv[i],"-o")) {
            ++i;
            outputFile = argv[i];
        } else {
            printf("Wrong input");
            exit(1);
        }
        i++;
    }
}
void getSize(int &size, char *file) {
    FILE *fp;
    fp = fopen(file,"r");
    if(fp == NULL) {
        perror("Error opening File\n");
        exit(1);
    }

    if(fscanf(fp,"%d",&size)==EOF) {
        printf("Error reading file\n");
        exit(1);
    }
    fclose(fp);
}
void readFromFile(int &size,float *v, char *file) {
    FILE *fp;
    fp = fopen(file,"r");
    if(fp == NULL) {
        printf("Error opening File %s\n",file);
        exit(1);
    }

    if(fscanf(fp,"%d",&size)==EOF) {
        printf("Error reading file\n");
        exit(1);
    }
    int i=0;
    float t;
    while(i < size) {
        if(fscanf(fp,"%f",&t)==EOF) {
            printf("Error reading file\n");
            exit(1);
        }
        v[i++]=t;
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    int ii;
    float *hostInput;      // input list
    float *hostOutput;     // output list
    float *deviceInput;
    float *deviceOutput;
    int numInputElements;  // number of elements in the input list
    int numOutputElements; // number of elements in the output list
    float *solution;

    // Read arguments and input files
    parseInput(argc,argv);

    // Read input from data
    getSize(numInputElements,inputFile);
    //numInputElements <<= 10;
    hostInput = (float*) calloc(numInputElements, sizeof(float));
    //numInputElements >>= 10;
    readFromFile(numInputElements,hostInput,inputFile);
    //numInputElements <<= 10;

    int opsz;
    getSize(opsz,outputFile);
    solution = (float*) calloc(opsz, sizeof(float));

    readFromFile(opsz,solution,outputFile);

    // ============== Assumes output element per block ==============
    numOutputElements = numInputElements / (BLOCK_SIZE << 1);
    if (numInputElements % (BLOCK_SIZE << 1)) {
        numOutputElements++;
    }
    hostOutput = (float *)calloc(numOutputElements, sizeof(float));

    if (numOutputElements > 512) {
        numOutputElements = 512;
    }
    // ==============================================================

    // Allocate GPU memory here
    cudaMalloc((float **) &deviceInput, numInputElements*sizeof(float));
    cudaMalloc((float **) &deviceOutput, numOutputElements*sizeof(float));
    cudaMemcpy(deviceInput, hostInput, numInputElements*sizeof(float), cudaMemcpyHostToDevice);

    // ====================== Initialize timer ======================
    cudaEvent_t start,stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    // Launch the GPU Kernel here, you may want multiple implementations to compare
    total_reduction<<<numOutputElements, BLOCK_SIZE>>>(deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();

    // Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, numOutputElements*sizeof(float), cudaMemcpyDeviceToHost);

    // Reduce any remaining output on host
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time,start, stop);
    // ==============================================================

    // Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Check solution
    if(solution[0] == hostOutput[0]) {
        printf("The operation was successful \n");
        printf("Time:                      %2.6f \n",elapsed_time);
    } else {
        printf("The operation failed \n");
    }
    printf("Expected sum:              %0.0f \n", solution[0]);
    printf("Computed sum:              %0.0f \n", hostOutput[0]);
    printf("Number of input elements:  %i \n", numInputElements);
    printf("Number of output elements: %i \n \n", numOutputElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
