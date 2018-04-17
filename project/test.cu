#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuComplex.h>
#include <fstream>

#include "Log.h"
#include "scatter.h"
#include "iohandler.h"

__global__
void test_kernel(int* image, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size)
        image[idx] = idx;
}

void test_schedule(JobScheduler* scheduler)
{
    int totalRequiredMemory = 100 * 100 * sizeof(int);
    Job* job = scheduler->addJob();
    auto jobLambda = [=] (cudaStream_t stream)
    {
        int* toDo;
        cudaMalloc((int**)&toDo,totalRequiredMemory);
        cudaMemset((void*)toDo,0,totalRequiredMemory);
        dim3 blocks(100,100);
        dim3 threads(100,100);

        job->registerCleanup([=] () 
        {
            LOG_DEBUG("Test cleanup called");
            int *h_result = (int*)malloc(totalRequiredMemory);
            cudaMemcpy(h_result,toDo,totalRequiredMemory,cudaMemcpyDeviceToHost);

            //cudaStreamSynchronize(stream);

            std::fstream out("Dump.bin",std::ios::out | std::ios::binary); 
            out.write((char*)h_result,totalRequiredMemory);
            out.close();

            cudaFree(toDo);

        });

        test_kernel<<<threads,blocks,0,stream>>>(toDo,100*100);
        cudaStreamAddCallback(stream,&Job::cudaCb,(void*)job,0);
    };

    job->setupJob(jobLambda,totalRequiredMemory,totalRequiredMemory);
    job->queue(); 
}