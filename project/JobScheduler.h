#ifndef __JobScheduler_h__
#define __JobScheduler_h__

#include "Job.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <condition_variable>

class JobScheduler 
{
    public:
    JobScheduler(int memoryHighWaterMark,int maxJobs);
    ~JobScheduler();

    Job* addJob();
    void queueUpJob(Job*);
    void jobDone(Job*);

    void waitUntilDone();

    private:

        void _checkIfCanRunJob();
        int _maxJobs;
        int _memoryHighWater;
        int _currentlyRunningJobs;
        //Probably a heap is better
        std::vector<Job*> _jobs;
        std::condition_variable _waitCv;
};


#endif