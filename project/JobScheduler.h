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

    private:

        size_t memoryAvailable() const;
        uint64_t highWaterMark() const;
        void _checkIfCanRunJob();
        int _maxJobs;
        uint64_t _currentMemoryUsage;
        int _currentlyRunningJobs;
        //Probably a heap is better
        std::vector<Job*> _jobs;
        std::condition_variable _waitCv;
        ThreadPool _threadPool;
};


#endif