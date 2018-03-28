#ifndef __Job_h__
#define __Job_h__

#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>


class JobScheduler;

class Job 
{
    public:
        ~Job();
        cudaStream_t& getStream();
        void setupJob(std::function<void (cudaStream_t&)> func,uint64_t requiredMemory,const std::string& path);
        void queue();
        void execute();
    private: 
        Job();

        cudaStream_t _stream;
        uint64_t _requiredBytes;
        JobScheduler* _scheduler;
        std::string _outputPath;
        std::function<void (cudaStream_t&)> _executionLambda;
        /* Speciailize the Job here a bit, we can always make this a lambda,
        but for now we always know we intend to read out a certain sized buffer
        and then write that out to the result location */


        

    friend class JobScheduler;

};

#endif