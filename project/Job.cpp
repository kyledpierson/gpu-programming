#include "Job.h"
#include <iostream>
#include "JobScheduler.h"
#include "Log.h"
#include "iohandler.h"


Job::Job()
    : _id(Job::GenerateGuid())
    , _running(false)
    , _done(false)
    , _lastBytes(0)
    , _bytesProcessed(0)
{
    cudaStreamCreate(&_stream);
}

void Job::setDone()
{
    _done = true;
}

Job::~Job()
{
    cudaStreamDestroy(_stream);
}
cudaStream_t& Job::getStream()
{
    return _stream;
}
bool Job::isReady() const
{
    if(!_running && _stages.size() > 0)
    {
        return true;
    }
    return false;
}

void Job::execute() //exeecutes a stage
{
    LOG_DEBUG(std::string("Executing Job Stage with ID: ") + _id);
    //TODO: Lambda?
    std::unique_lock<std::mutex> lock(_stageMutex);
    if(_stages.size() > 0)
    {
        startTimer();
        _running = true;
        Stage& stage = _stages.front();
        stage.lambda(_stream);
        _lastBytes = stage.requiredBytes;
        _stages.pop_front();
        _running = false;
    }
}
uint64_t Job::requiredMemory() const
{
    std::unique_lock<std::mutex> slock(_stageMutex);
    if(_stages.size() > 0)
    {
        return _stages.front().requiredBytes;
    }
    return 0;
}

void Job::addStage(std::function<void (cudaStream_t&)> func,uint64_t requiredMemory,uint64_t inputSize)
{
    std::unique_lock<std::mutex> lock(_stageMutex);
    LOG_DEBUG(std::string("Added Job stage with ID: ") + _id);
    _stages.push_back(Stage(func,requiredMemory,inputSize));
    _bytesProcessed += inputSize;
}

Job::Stage::Stage(std::function<void (cudaStream_t&)> func,uint64_t requiredBytes, uint64_t inputSize)
    : lambda(func)
    , requiredBytes(requiredBytes)
    , inputSize(inputSize)
{

}
void Job::queue()
{
    _scheduler->queueUpJob(this);
}

void Job::addFree(void* toFree,bool cuda)
{
    _toFree.insert(std::make_pair(cuda,toFree));
}

void Job::FreeMemory()
{
    for(auto memPair : _toFree)
    {
        if(memPair.first)
            cudaFree(memPair.second);
        else
            free(memPair.second);
    }
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
    stopTimer();
    //IMPORTANT: Stream callbacks can't call CUDA calls!
    //Need to be tricksy about this, let's go ahead and 
    //Schedule on a thread pool for this job to be run 
    //http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-callbacks
    //It must do some checking to which thread is calling it.

    //At this point, could technically move this all into cudaCb,
    //possibly do that in the future
    if(_done)
    {
        _scheduler->queueCallback(this,[=] () { 
            this->startTimer();
            this->_cleanupFunc(); 
            this->stopTimer();
            _scheduler->jobDone(this);  //calls checkIfCanRunJob
        });
    } 
    else 
    {
        //Just let it run it's own scheduler.
        _scheduler->checkIfCanRunJob();
    }

}
int64_t Job::totalMs() const
{
    int64_t total = 0;
    for( auto tp : _timePeriods)
    {
        if(tp.complete())
            total += tp.ms();
        else
        {
            LOG_DEBUG("Tried to get total MS for non-completed timer");
            LOG_DEBUG(std::string("Started: ") + (tp.started() ? "yes" : "no"));
            LOG_DEBUG(std::string("ended : ") + (tp.ended() ? "yes" : "no"));
        }
    }
    return total;
}

std::string Job::GenerateGuid()
{
    char strUuid[512];

    sprintf(strUuid, "%x%x-%x-%x-%x-%x%x%x", 
    rand(), rand(),                 // Generates a 64-bit Hex number
    rand(),                         // Generates a 32-bit Hex number
    ((rand() & 0x0fff) | 0x4000),   // Generates a 32-bit Hex number of the form 4xxx (4 indicates the UUID version)
    rand() % 0x3fff + 0x8000,       // Generates a 32-bit Hex number in the range [0x8000, 0xbfff]
    rand(), rand(), rand());        // Generates a 96-bit Hex number
    return std::string(strUuid);
}
void Job::startTimer()
{
    _timePeriods.push_back(TimePeriod());
    _timePeriods.back().start();
    //LOG_DEBUG(std::string("Start timer for ") + _id + std::string(" Idx: ") + std::to_string(_timePeriods.size()-1));
}
void Job::stopTimer()
{
    //Finish the first available timer
    int idx = 0;
    for(auto& timer : _timePeriods)
    {
        if(!timer.ended())
        {
            timer.stop();
//            LOG_DEBUG(std::string("Stop timer for ") + _id + std::string(" idx ") + std::to_string(idx));
            return;
        }
        idx++;
    }

}
int64_t Job::bytesPerMs() const
{
    return bytesProcessed() / totalMs();
}