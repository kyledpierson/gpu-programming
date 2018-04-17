#include "JobScheduler.h"
#include <mutex>
#include "Log.h"


JobScheduler::JobScheduler(int maxJobs)
    : _maxJobs(maxJobs)
    , _currentlyRunningJobs(0)
    , _currentMemoryUsage(0)
    , _threadPool(5)
{

}

JobScheduler::~JobScheduler()
{
    LOG_DEBUG("Finishing up thread pool jobs in scheduler...")
    _threadPool.finishAllJobs();
}

Job* JobScheduler::addJob()
{
    auto job = new Job();
    job->_scheduler = this;
    return job;
}

void JobScheduler::queueUpJob(Job* job)
{
    _jobs.push_back(job);
    _checkIfCanRunJob();
}

void JobScheduler::_checkIfCanRunJob()
{
    std::unique_lock<std::mutex> lock(_jobLock);
    /* This function will look through all of the current jobs and determine if
    any of them will fit onto the GPU (based on memory high water mark).
    */
    //For now, just run the jobs
    for(auto it = _jobs.begin(); it != _jobs.end(); )
    {
        //0 denotes no job limit
        if(_maxJobs > 0 && _maxJobs <= _currentlyRunningJobs)
        {
            LOG_DEBUG("Already at the maximum number of jobs...");
            it = _jobs.end();; //can't do any more....
            break;
        }
        if((*it)->requiredMemory() + _currentMemoryUsage < highWaterMark())
        {
            //found a job that will work
            (*it)->execute();
            _currentMemoryUsage += (*it)->requiredMemory();
            _currentlyRunningJobs++;
            LOG_DEBUG(std::string("Found new job I can run that will require ") + std::to_string((*it)->requiredMemory() / 1024 / 1024) + " MB I have " + std::to_string(highWaterMark() / 1024/1024) + " MB avail");;
            if(it != _jobs.end())
            it = _jobs.erase(it);
        }
        else
        {
            it++;
        }
    }
}

void JobScheduler::waitUntilDone()
{
    std::mutex m;
    std::unique_lock<std::mutex> lk(m);
    while(_currentlyRunningJobs > 0 || _jobs.size() > 0)
        _waitCv.wait(lk);
    LOG_DEBUG("Falling out of the wait, no more jobs to process");

}
void JobScheduler::jobDone(Job* job)
{
    //TODO: Probably want to mutex this
    _currentlyRunningJobs--;
    _currentMemoryUsage -= job->requiredMemory();
    _checkIfCanRunJob();
    _waitCv.notify_all();

}
size_t JobScheduler::memoryAvailable() const
{
    size_t freem, total;
    //CUDA_SAFE_CALL(cudaMemGetInfo(&freem,&total));
    cudaMemGetInfo(&freem,&total);

    LOG_DEBUG(std::string("Total memory free: ") + std::to_string(freem));
    return freem;
}
uint64_t JobScheduler::highWaterMark() const
{
    return (uint64_t)(.85 * memoryAvailable());
}

void JobScheduler::queueCallback(Job* job, std::function<void ()> func)
{
    LOG_DEBUG("Scheduling callback on our own thread pool...");
    _threadPool.queueJob(
    [=] () {
        func();
        jobDone(job);
    }

    );
}