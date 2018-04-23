#ifndef __Job_h__
#define __Job_h__

#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <set>
#include <vector>
#include <atomic>
#include <map>
#include <mutex>
#include <chrono>
#include <deque>
#include "Log.h"


class JobScheduler;

class Job
{
    public:
        ~Job();
        static std::string GenerateGuid();
        cudaStream_t& getStream();
        void addStage(std::function<void (cudaStream_t&)> func,uint64_t requiredMemory,uint64_t inputSize);
        void queue();
        void execute();
        void addFree(int stage,void*,bool);
        void registerCleanup(std::function<void ()> clean) { _cleanupFunc = clean; }
        uint64_t requiredMemory() const;
        void FreeMemory(int stage);
        bool isReady() const;
        std::string id() const { return _id; }

        void startTimer();
        void stopTimer();
        void setDone();
        uint64_t lastBytes() const { return _lastBytes; }

        static void CUDART_CB cudaCb(cudaStream_t stream, cudaError_t status, void *userData);
        struct MemoryCbData {
            Job* job;
            int stage;
            bool setDone;
        };
        static void CUDART_CB memoryCb(cudaStream_t stream, cudaError_t status, void* userData);

    private:
        std::string _id;
        std::atomic_bool _running;
        std::atomic_bool _done;
        mutable std::mutex _stageMutex;
        Job();
        void _internalCb();

        struct ResultInfo
        {
            int64_t offset;
            void* source;
            uint64_t size;
        };


        cudaStream_t _stream;
        std::map<int,std::set<std::pair<bool,void*> > > _toFree;
        uint64_t _requiredBytes;
        uint64_t _bytesProcessed;
        JobScheduler* _scheduler;

        struct Stage
        {
            Stage(std::function<void (cudaStream_t&)>,uint64_t requiredBytes, uint64_t inputSize);
            std::function<void (cudaStream_t&)> lambda;
            uint64_t requiredBytes;
            uint64_t inputSize;
        };



        std::deque<Stage> _stages;
        std::function<void ()> _cleanupFunc;
        uint64_t _lastBytes;

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
        int64_t bytesProcessed() const { return _bytesProcessed; }


    friend class JobScheduler;

};

#endif
