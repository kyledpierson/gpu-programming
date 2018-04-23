#ifndef __JobScheduler_h__
#define __JobScheduler_h__

#include "Job.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <condition_variable>
#include "ThreadPool.h"

class JobScheduler
{
    public:
    JobScheduler(int maxJobs);
    ~JobScheduler();

    Job* addJob();
    void queueUpJob(Job*);
    void jobDone(Job*);
    void queueCallback(Job* job, std::function<void ()> func);

    void waitUntilDone();
    void checkIfCanRunJob();
    void ping();
    int currentJobs() { return _currentlyRunningJobs; }

    private:

        size_t memoryAvailable() const;
        uint64_t highWaterMark() const;
        int _maxJobs;
        uint64_t _currentMemoryUsage;
        int _currentlyRunningJobs;
        //Probably a heap is better
        std::vector<std::unique_ptr<Job> > _jobs;
        std::condition_variable _waitCv;
        ThreadPool _threadPool;
        std::mutex _jobLock;
        uint64_t _totalUsedMs;
        uint64_t _totalBytesDone;
        uint64_t _totalFilesProcessed;
};


#endif
