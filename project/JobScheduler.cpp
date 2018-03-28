#include "JobScheduler.h"


JobScheduler::JobScheduler(int memoryHighWater,int maxJobs)
    : _memoryHighWater(memoryHighWater)
    , _maxJobs(maxJobs)
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
        job->execute();
    }

}
