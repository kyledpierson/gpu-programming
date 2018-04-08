#ifndef __THREADPOOL_H__
#define __THREADPOOL_H__

#include <thread>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>

typedef std::function<void()> vFunc;

class ThreadPool
{
    private:
        //-----------
        /* Members */
        //-----------

        int _threadCount;
        std::vector<std::unique_ptr<std::thread> > _workers;
        volatile bool _run;
        // Don't accept new jobs
        volatile bool _acceptNewJobs;
        std::mutex _jobLock;
        std::condition_variable _notifyJob;
        std::queue<vFunc> _jobQueue;

        vFunc _nopJob;


        //-------------
        /* Functions */
        //------------

        void _init();
        void _loop();
        vFunc _getJob(bool&);


    public:
        ThreadPool(int threads);
        virtual ~ThreadPool();

        void queueJob(vFunc job);

        // Call this to gracefully shut down threads (blocks)
        void finishAllJobs();
    private:
        // Clean up function; don't mess with this
        // Called by destructor and finishAllJobs();
        void stopPool();
};


#endif
