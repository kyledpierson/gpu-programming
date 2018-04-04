#ifndef __Job_h__
#define __Job_h__

#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <set>
#include <vector>


class JobScheduler;

class Job
{
    public:
        ~Job();
        static std::string GenerateGuid();
        cudaStream_t& getStream();
        void setupJob(std::function<void (cudaStream_t&)> func,uint64_t requiredMemory,const std::string& path);
        void queue();
        void execute();
        void addFree(void*,bool);
        void addResultInfo(void* res, uint64_t size, uint64_t offset)
        {
            ResultInfo rinfo;
            rinfo.offset = offset;
            rinfo.source = res;
            rinfo.size = size;
            _results.push_back(rinfo);
        }
        uint64_t requiredMemory() const { return _requiredBytes; }

        static void CUDART_CB cudaCb(cudaStream_t stream, cudaError_t status, void *userData);

    private:
        std::string _id;
        Job();
        void _internalCb();

        struct ResultInfo
        {
            int64_t offset;
            void* source;
            uint64_t size;
        };

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
        std::vector<ResultInfo> _results;
        /* Speciailize the Job here a bit, we can always make this a lambda,
        but for now we always know we intend to read out a certain sized buffer
        and then write that out to the result location */




    friend class JobScheduler;

};

#endif
