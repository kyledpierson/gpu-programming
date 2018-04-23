#include "Log.h"

#include <fstream>


namespace Log {
    static std::string logFile;
    static std::ofstream logHandle;

    void log_file(const std::string& line)
    {
        if(logFile.size() > 0)
        {
            logHandle.write(std::string(line + "\n").c_str(),line.size() + 1);
        }
        else
        {
            log("ERROR","No log file opened to log to");
        }
    }
    void initLogFile(const std::string& lFile)
    {
        logFile = lFile;
        logHandle.open(logFile);
    }
    void closeLog()
    {
        if(logFile.size() > 0)
            logHandle.close();
    }

    void log(const std::string& level, const std::string& msg)
    {
        if(level != "DEBUG")
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
