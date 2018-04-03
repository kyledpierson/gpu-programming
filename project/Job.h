#ifndef __Job_h__
#define __Job_h__

#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <set>


class JobScheduler;

class Job 
{
    public:
        ~Job();
        cudaStream_t& getStream();
        void setupJob(std::function<void (cudaStream_t&)> func,uint64_t requiredMemory,const std::string& path);
        void queue();
        void execute();
        void addFree(void*,bool);
        void addResultInfo(void* res, uint64_t size, uint64_t xSize, uint64_t ySize) { _resultFrom = res; _resultSize = size; _resultXDim = xSize; _resultYDim = ySize;}
        uint64_t requiredMemory() const { return _requiredBytes; }

        static void CUDART_CB cudaCb(cudaStream_t stream, cudaError_t status, void *userData);

    private: 
        Job();
        void _internalCb();

        cudaStream_t _stream;
        std::set<std::pair<bool,void*> > _toFree;
        uint64_t _resultSize;
        uint64_t _requiredBytes;
        uint64_t _resultXDim;
        uint64_t _resultYDim;
        void*    _resultFrom;
        JobScheduler* _scheduler;
        std::string _outputPath;
        std::function<void (cudaStream_t&)> _executionLambda;
        /* Speciailize the Job here a bit, we can always make this a lambda,
        but for now we always know we intend to read out a certain sized buffer
        and then write that out to the result location */


        

    friend class JobScheduler;

};

#endif