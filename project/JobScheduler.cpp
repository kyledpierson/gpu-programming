#include "JobScheduler.h"
#include <mutex>
#include "Log.h"


JobScheduler::JobScheduler(int maxJobs)
    : _maxJobs(maxJobs)
    , _currentlyRunningJobs(0)
    , _currentMemoryUsage(0)
    , _threadPool(5)
    , _totalUsedMs(0)
    , _totalBytesDone(0)
    , _totalFilesProcessed(0)
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
    checkIfCanRunJob();
}

void JobScheduler::checkIfCanRunJob()
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
        if((*it)->isReady() && (*it)->requiredMemory() + _currentMemoryUsage < highWaterMark())
        {
            //found a job that will work
            LOG_DEBUG(std::string("Found new job I can run that will require ") + std::to_string((*it)->requiredMemory() / 1024 / 1024) + " MB I have " + std::to_string(highWaterMark() / 1024/1024) + " MB avail");;
            _currentMemoryUsage += (*it)->requiredMemory();
            (*it)->execute();
            _currentlyRunningJobs++;
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
    LOG_DEBUG("----------------------")
    LOG_DEBUG("-- FINISH --");
    LOG_DEBUG("----------------------")
    LOG_DEBUG("Completed " + std::to_string(_totalFilesProcessed) + " files");
    LOG_DEBUG("Total Bytes Processed: " + std::to_string(_totalBytesDone/1024/1024) + " MB");
    LOG_DEBUG(std::string("KB/S: ") + std::to_string((_totalBytesDone /1024) / (_totalUsedMs / 1000)))

}
void JobScheduler::jobDone(Job* job)
{
    //TODO: Probably want to mutex this
    _currentlyRunningJobs--;
    _currentMemoryUsage -= job->lastBytes();
    auto totalTime = job->totalMs();
    LOG_DEBUG(std::string("Job took " ) + std::to_string(totalTime) + " MS");
    LOG_DEBUG(std::string("KBytes/MS for this job: ") + std::to_string(job->bytesPerMs()/(1024)));
    LOG_DEBUG(std::string("Bytes processed: " + std::to_string(job->bytesProcessed())));
    _totalUsedMs += totalTime;
    _totalBytesDone += job->bytesProcessed();
    _totalFilesProcessed++;
    delete job;

    checkIfCanRunJob();
    _waitCv.notify_all();

}
size_t JobScheduler::memoryAvailable() const
{
    size_t freem, total;
    //CUDA_SAFE_CALL(cudaMemGetInfo(&freem,&total));
    cudaMemGetInfo(&freem,&total);

    //LOG_DEBUG(std::string("Total memory free: ") + std::to_string(freem));
    return freem;
}
uint64_t JobScheduler::highWaterMark() const
{
    return (uint64_t)(.85 * memoryAvailable());
}

void JobScheduler::queueCallback(Job* job, std::function<void ()> func)
{
    LOG_DEBUG("Scheduling callback on our own thread pool...");
    _threadPool.queueJob(func);
}