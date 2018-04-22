#ifndef __Log_h__
#define __Log_h__
/* Just some utility functions for logging */

#include <iostream>

#define LOG_DEBUG(str) Log::log("DEBUG",str);
#define LOG_FILE(str) Log::log_file(str)

namespace Log {
    void log(const std::string& level, const std::string&);
    void log_file(const std::string& line);
    void initLogFile(const std::string& logFile);
    void closeLog();
    std::string timeString();
};


#define CUDA_SAFE_CALL(call)                                          \
{                                                                     \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    } \
}

#endif