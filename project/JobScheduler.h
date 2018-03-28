#ifndef __JobScheduler_h__
#define __JobScheduler_h__

#include "Job.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

class JobScheduler 
{
    public:
    JobScheduler(int memoryHighWaterMark,int maxJobs);
    ~JobScheduler();

    Job* addJob();
    void queueUpJob(Job*);

    private:

        void _checkIfCanRunJob();
        int _maxJobs;
        int _memoryHighWater;
        //Probably a heap is better
        std::vector<Job*> _jobs;
};


#endif