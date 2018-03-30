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
    _requiredBytes = requiredMemory;
    _outputPath = path;
}
void Job::queue()
{
    _scheduler->queueUpJob(this);
}

void Job::addFree(void* toFree)
{
    _toFree.insert(toFree);

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
    int *result = (int*) mem_check(malloc(_resultSize));
    cudaMemcpy(result, _resultFrom, _resultSize, cudaMemcpyDeviceToHost);
    for(void* item : _toFree)
    {
        cudaFree(item);
    }
    //Once we know, write out the result file
    //TODO: Temporary, just dump the file
    LOG_DEBUG(std::string("Writing result to ") + _outputPath);
    write_ppm(const_cast<char*>(_outputPath.c_str()),_resultXDim,_resultYDim,255,result);
    free(result);
    _scheduler->jobDone(this);

}