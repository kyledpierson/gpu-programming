#ifndef __Job_h__
#define __Job_h__

#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <set>
#include <vector>
#include <chrono>
#include "Log.h"


class JobScheduler;

class Job
{
    public:
        ~Job();
        static std::string GenerateGuid();
        cudaStream_t& getStream();
        void setupJob(std::function<void (cudaStream_t&)> func,uint64_t requiredMemory,uint64_t inputSize);
        void queue();
        void execute();
        void addFree(void*,bool);
        void registerCleanup(std::function<void ()> clean) { _cleanupFunc = clean; }
        uint64_t requiredMemory() const { return _requiredBytes; }
        void FreeMemory();

        void startTimer();
        void stopTimer();

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
        uint64_t _requiredBytes;
        uint64_t _inputSize;
        JobScheduler* _scheduler;
        std::function<void (cudaStream_t&)> _executionLambda;
        std::function<void ()> _cleanupFunc;

        struct TimePeriod 
        {
            std::chrono::high_resolution_clock::time_point _start;
            std::chrono::high_resolution_clock::time_point _end;
            bool _started,_ended; //doesn't seem to work with default values
            // for checking if it's done...
            TimePeriod()
            : _started(false) , _ended(false) {}
            void start()
            {
                _start = std::chrono::high_resolution_clock::now();
                _started = true;
            }
            void stop()
            {
                _end = std::chrono::high_resolution_clock::now();
                _ended = true;
            }
            int64_t ms() const
            {
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start);
                return ms.count();
            }

            bool started() const 
            {
                return _started;
                //return _start != std::chrono::high_resolution_clock::time_point();
            }
            bool ended() const
            {
                return _ended;
                //return _end != std::chrono::high_resolution_clock::time_point();
            }
            bool complete() const 
            {
                return started() && ended();
            }
        };

        std::vector<TimePeriod> _timePeriods;
        int64_t totalMs() const;
        int64_t bytesPerMs() const;


    friend class JobScheduler;

};

#endif
