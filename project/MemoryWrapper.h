#ifndef __MemoryWrapper_h__
#define __MemoryWrapper_h__

#include "Log.h"
#include "ThreadPool.h"
#include <mutex>


//CUDA does not allow you to call cudaMalloc and cudaFree from separate threads!!!!
// You MUST use the same thread (that's stupid!)

class MemoryWrapper {
    public:
        MemoryWrapper();
        static void init();
        static void* malloc(uint64_t size);
        static void free(void*);
    private:
        ThreadPool _pool;
        std::mutex _mutex;

};

static MemoryWrapper mw;

#endif
