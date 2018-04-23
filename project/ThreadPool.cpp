#include <iostream>

#include "ThreadPool.h"


ThreadPool::ThreadPool(int threads) : _threadCount(threads),_run(true),_acceptNewJobs(true),_nopJob([] () {})
{
    _init();
}

ThreadPool::~ThreadPool()
{
    stopPool();
}

void ThreadPool::queueJob(vFunc job)
{
    if(!_acceptNewJobs) return;

    std::unique_lock<std::mutex> lock(_jobLock);
    _jobQueue.push(job);
    _notifyJob.notify_one();
}


/* I would have loved to not use a specialNop, and instead have the
 * caller of this function be able to differentiate the specialNop
 * from any other returned function. However, that proved
 * fruitless as _nopJob can not be identified. If I changed the return
 * type to a pointer it could, as it would be able to check the pointer
 * address of the std::function(), however, that would mean I would have to
 * new or otherwise get the memory address of each of the jobs, and as
 * they are passed in via queueJob, those would have to be pointers already
 * (which would then beg the question of ownership), or get converted to
 * pointers either via a copy (Stack -> heap)or by handing out the address in the
 * queue, and keeping it around in memory (somewhere) which would be messy. This unfortunately
 * seems to be the best way to allow for the finishAllJobs functionality,
 * and luckily it's within the guts of the thread pool so
 * the caller never has to know it's dirty secret. */
vFunc ThreadPool::_getJob(bool& specialNop)
{
    //Grab the lock
    specialNop = false;
    std::unique_lock<std::mutex> lock(_jobLock);

    while(_run)
    {
        if(!_jobQueue.empty())
        {
            vFunc func = _jobQueue.front();
            _jobQueue.pop();
            return std::move(func);
        }
        //If we are not accepting jobs and the queue is empty, just bail out
        if(!_acceptNewJobs) { specialNop = true; return _nopJob; }

        _notifyJob.wait(lock);
    }

    return _nopJob;
}

void ThreadPool::stopPool()
{
    if(!_run) return;
    {
        std::unique_lock<std::mutex> lock(_jobLock);
        _run = false;
        _acceptNewJobs = false;
        // Let's all the threads waiting in _getJob() know to check again,
        // which would force them to see the _acceptNewJobs as false, and bail.
        _notifyJob.notify_all();
    }

    for(std::unique_ptr<std::thread>& worker : _workers)
    {
        if(worker->joinable())
            worker->join();
    }
}

void ThreadPool::finishAllJobs()
{
    _acceptNewJobs = false;

    std::unique_lock<std::mutex> lock(_jobLock);
    while(!_jobQueue.empty())
        _notifyJob.wait(lock);


    /*
    for(std::unique_ptr<std::thread>& worker : _workers)
    {
        if(worker->joinable())
            worker->join();
    }
    */

    //stopPool();
}


void ThreadPool::_init()
{
    for(int i = 0; i < _threadCount;i++)
    {
        _workers.push_back( std::unique_ptr<std::thread>(new std::thread(&ThreadPool::_loop,this)) );
    }

}

void ThreadPool::_loop()
{
    for(;;)
    {
        if(!_run) return;
        //The waiting occurs in _getJob();
        bool specialNop = false;
        vFunc job = _getJob(specialNop);
        if(specialNop && !_acceptNewJobs)
        {
            //We got the NOP job, which is fine in the case of shutdown
            // but if we got the NOP and we are not accepting new jobs
            // it means that we want to finish this thread.
            _notifyJob.notify_all();
            return;
        }
        try
        {
            job();
        }
        catch(...)
        {
            //There was an error in that function call, but we don't handle that shit.
            std::cerr << "Error running job..." << std::endl;
        }
    }
}

