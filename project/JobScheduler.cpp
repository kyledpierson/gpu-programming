#include "JobScheduler.h"
#include <mutex>
#include "Log.h"


JobScheduler::JobScheduler(int memoryHighWater,int maxJobs)
    : _memoryHighWater(memoryHighWater)
    , _maxJobs(maxJobs)
    , _currentlyRunningJobs(0)
{

}

JobScheduler::~JobScheduler()
{

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
    /* This function will look through all of the current jobs and determine if
    any of them will fit onto the GPU (based on memory high water mark).
    */
    //For now, just run the jobs
    for(auto job : _jobs)
    {
        _currentlyRunningJobs++;
        job->execute();
    }
    _jobs.clear();

}

void JobScheduler::waitUntilDone()
{
    std::mutex m;
    std::unique_lock<std::mutex> lk(m);
    while(_currentlyRunningJobs > 0 || _jobs.size() > 0)
        _waitCv.wait(lk);
    LOG_DEBUG("Falling out of the wait, no more jobs to process");

}
//TODO: Convert this to use unique Guids as job ids, not the pointer
void JobScheduler::jobDone(Job* job)
{
    _currentlyRunningJobs--;
    _checkIfCanRunJob();
    _waitCv.notify_all();

}