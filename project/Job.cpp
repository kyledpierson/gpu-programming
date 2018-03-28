#include "Job.h"
#include <iostream>
#include "JobScheduler.h"
#include "Log.h"

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
    //TODO: get a real logging set of macros, at the least
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