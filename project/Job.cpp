#include "Job.h"
#include <iostream>
#include "JobScheduler.h"
#include "Log.h"
#include "iohandler.h"


Job::Job()
    : _id(Job::GenerateGuid())
{
    cudaStreamCreate(&_stream);
}

Job::~Job()
{
    cudaStreamDestroy(_stream);
}
cudaStream_t& Job::getStream()
{
    return _stream;
}

void Job::execute()
{
    LOG_DEBUG(std::string("Executing Job with ID: ") + _id);
    _executionLambda(_stream);
}

void Job::setupJob(std::function<void (cudaStream_t&)> func,uint64_t requiredMemory)
{
    _executionLambda = func;
    //LOG_DEBUG(std::string("Setting up function to require ") + std::to_string(requiredMemory) + " bytes of memory and output to " + path);
    _requiredBytes = requiredMemory;
}
void Job::queue()
{
    _scheduler->queueUpJob(this);
}

void Job::addFree(void* toFree,bool cuda)
{
    _toFree.insert(std::make_pair(cuda,toFree));
}


__host__
void CUDART_CB Job::cudaCb(cudaStream_t stream, cudaError_t status, void *userData)
{
    //LOG_DEBUG("Static callback called for job");
    Job* self = static_cast<Job*>(userData);
    self->_internalCb();
}


__host__
void Job::_internalCb()
{
    //Some magic thread pool is calling this callback
    LOG_DEBUG(std::string("Calling job call back for id: ") + _id);
    /*
    uint64_t resultSize = 0;
    //cudaMemcpy(result, _resultFrom, _resultSize, cudaMemcpyDeviceToHost);
    for(auto res : _results)
    {
        resultSize += res.size;
    }
    LOG_DEBUG(std::string("Total result size: ") + std::to_string(resultSize));
    int *result = (int*) mem_check(malloc(resultSize));
    if(result == nullptr)
    {
        LOG_DEBUG("UNABLE TO MALLOC ENOUGH MEMORY");
    }
    */

/*
    for(auto res : _results)
    {
        LOG_DEBUG(std::string("Copying ") + std::to_string(res.size) + " to offset " + std::to_string(res.offset/1024/1024));
        LOG_DEBUG("Copying from address " + std::to_string((uint64_t)res.source));
        CUDA_SAFE_CALL(cudaMemcpy(result + res.offset,res.source,res.size,cudaMemcpyDeviceToHost));
    }
    */

/*
    for(std::pair<bool,void*> item : _toFree)
    {
        if(item.first)
            cudaFree(item.second);
        else
            free(item.second);
    }
    */
    //Once we know, write out the result file
    _cleanupFunc();
    _scheduler->jobDone(this);

}

std::string Job::GenerateGuid()
{
    char strUuid[512];
    srand(time(NULL));

    sprintf(strUuid, "%x%x-%x-%x-%x-%x%x%x", 
    rand(), rand(),                 // Generates a 64-bit Hex number
    rand(),                         // Generates a 32-bit Hex number
    ((rand() & 0x0fff) | 0x4000),   // Generates a 32-bit Hex number of the form 4xxx (4 indicates the UUID version)
    rand() % 0x3fff + 0x8000,       // Generates a 32-bit Hex number in the range [0x8000, 0xbfff]
    rand(), rand(), rand());        // Generates a 96-bit Hex number
    return std::string(strUuid);
}