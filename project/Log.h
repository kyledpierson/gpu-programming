#ifndef __Log_h__
#define __Log_h__
/* Just some utility functions for logging */

#include <iostream>

#define LOG_DEBUG(str) Log::log("DEBUG",str);

namespace Log {
    void log(const std::string& level, const std::string&);
    std::string timeString();
};




#endif