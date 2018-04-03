#include "Job.h"
#include <iostream>
#include "JobScheduler.h"
#include "Log.h"
#include "iohandler.h"


Job::Job()
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
    LOG_DEBUG("Executing Job with cuda instructions");
    _executionLambda(_stream);
}

void Job::setupJob(std::function<void (cudaStream_t&)> func,uint64_t requiredMemory,const std::string& path)
{
    _executionLambda = func;
    LOG_DEBUG(std::string("Setting up function to require ") + std::to_string(requiredMemory) + " bytes of memory and output to " + path);
    _requiredBytes = requiredMemory;
    _outputPath = path;
}
void Job::queue()
{
    _scheduler->queueUpJob(this);
}

void Job::addFree(void* toFree,bool cuda)
{
    _toFree.insert(std::make_pair(cuda,toFree));
}


void CUDART_CB Job::cudaCb(cudaStream_t stream, cudaError_t status, void *userData)
{
    LOG_DEBUG("Static callback called for job");
    Job* self = static_cast<Job*>(userData);
    self->_internalCb();
}


void Job::_internalCb()
{
    //Some magic thread pool is calling this callback
    LOG_DEBUG("Calling specific job callback");
    uint64_t resultSize = 0;
    //cudaMemcpy(result, _resultFrom, _resultSize, cudaMemcpyDeviceToHost);
    for(auto res : _results)
    {
        resultSize += res.size;
    }
    LOG_DEBUG(std::string("Total result size: ") + std::to_string(resultSize/1024/1024));
    int *result = (int*) mem_check(malloc(resultSize));

    for(auto res : _results)
    {
        LOG_DEBUG(std::string("Copying ") + std::to_string(res.size/1024/1024) + " MB to offset " + std::to_string(res.offset/1024/1024));
        CUDA_SAFE_CALL(cudaMemcpy(result + res.offset,res.source,res.size,cudaMemcpyDeviceToHost));
    }

    for(std::pair<bool,void*> item : _toFree)
    {
        if(item.first)
            cudaFree(item.second);
        else
            free(item.second);
    }
    //Once we know, write out the result file
    LOG_DEBUG(std::string("Writing result to ") + _outputPath);
    write_ppm(const_cast<char*>(_outputPath.c_str()),_resultXDim,_resultYDim,255,result);
    free(result);
    _scheduler->jobDone(this);

}
