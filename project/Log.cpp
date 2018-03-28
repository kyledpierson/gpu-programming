#include "Log.h"

namespace Log {

    void log(const std::string& level, const std::string& msg)
    {
        std::cout << timeString() << " " << level << " -- " << msg << std::endl;
    }

    std::string timeString()
    {
        static time_t rawtime;
        static struct tm * timeinfo;

        time ( &rawtime );
        timeinfo = localtime ( &rawtime );
        std::string timeStr = std::string(asctime (timeinfo) );
        return timeStr.substr(0,timeStr.length()-1);
    }
}