#include "MemoryWrapper.h"

#include <mutex>
#include <condition_variable>
#include <cuda.h>

//MemoryWrapper mw;

MemoryWrapper::MemoryWrapper()
    : _pool(1)
{
}

void* MemoryWrapper::malloc(uint64_t size)
{
    std::unique_lock<std::mutex> mLock(mw._mutex);
    std::condition_variable cVar;
    bool done = false;
    void* ptr = nullptr;

    mw._pool.queueJob(
            [&cVar,&ptr,size,&done]
            {
                cudaMalloc(&ptr,size);
                cudaMemset(ptr,0,size);
                done = true;
                cVar.notify_one();
            });

    while(!done)
        cVar.wait(mLock);

    return ptr;


}

void MemoryWrapper::free(void* ptr)
{
    std::unique_lock<std::mutex> mLock(mw._mutex);
    std::condition_variable cVar;
    bool done = false;

    mw._pool.queueJob(
            [&cVar,ptr,&done]
            {
                cudaFree(ptr);
                done = true;
                cVar.notify_one();
            });

    while(!done)
        cVar.wait(mLock);

}
