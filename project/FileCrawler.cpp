#include "FileCrawler.h"

#include <dirent.h>
#include "Log.h"


FileCrawler::FileCrawler(const std::string& baseDir, const std::string& extension)
    : _base(baseDir)
    , _extension(extension)
{

}
std::vector<FileCrawler::CFile> FileCrawler::getAllPaths()
{
    return _foundFiles;
}

bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void FileCrawler::crawl(const std::string& currentDir)
{
    struct dirent *ent;
    std::string dirMe = currentDir.size() > 0 ? currentDir : _base;
    LOG_DEBUG("Starting crawl on " + dirMe)

    DIR* dir = opendir(dirMe.c_str());
    if(dir != nullptr)
    {
        ent = readdir(dir);
        if(ent == nullptr)
        {
            LOG_DEBUG("No files in directory?");
        }
        while(ent)
        {
            if(!ent->d_name || ent->d_name[0] == '.')
            {
             //   LOG_DEBUG("CHECK");
            }
            else
            {
                if(ent->d_type == DT_REG && has_suffix(std::string(ent->d_name),_extension))
                {
                    _foundFiles.push_back(CFile(std::string(dirMe) + std::string("/") + ent->d_name,std::string(ent->d_name)));
                    //LOG_DEBUG("Found: " + _foundPaths.back());
                }
                else if(ent->d_type == DT_DIR)
                {
                    LOG_DEBUG(std::string("Recurse down to ") + ent->d_name)
                    //Linux specific
                    crawl(dirMe + std::string("/") + ent->d_name);
                }
                else
                {
            //        LOG_DEBUG("Don't care about this entity");
                }
            }
            ent = readdir(dir);
        }
        closedir(dir);
    }
    else 
    {
        LOG_DEBUG(std::string("Failed to open: ") + currentDir);
    }
}