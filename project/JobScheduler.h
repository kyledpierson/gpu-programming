#ifndef __JobScheduler_h__
#define __JobScheduler_h__

#include "Job.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <condition_variable>
#include "ThreadPool.h"
#include "MemoryWrapper.h"

class JobScheduler
{
    public:
    JobScheduler(int maxJobs);
    ~JobScheduler();

    Job* addJob();
    void queueUpJob(Job*);
    void jobDone(Job*);
    void queueCallback(Job* job, std::function<void ()> func);
    void* cudaMalloc(uint64_t );
    void cudaFree(void* ptr);

    void waitUntilDone();
    void checkIfCanRunJob();
    void ping();

    private:

        size_t memoryAvailable() const;
        uint64_t highWaterMark() const;
        int _maxJobs;
        uint64_t _currentMemoryUsage;
        int _currentlyRunningJobs;
        //Probably a heap is better
        std::vector<Job*> _jobs;
        std::condition_variable _waitCv;
        ThreadPool _threadPool;
        MemoryWrapper _memman;
        std::mutex _jobLock;
        uint64_t _totalUsedMs;
        uint64_t _totalBytesDone;
        uint64_t _totalFilesProcessed;
};


#endif
