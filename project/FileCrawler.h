#ifndef __File__Crawler_h__
#define __File__Crawler_h__

#include <iostream>
#include <vector>

class FileCrawler
{
    public:

    struct CFile
    {
        std::string _fullPath;
        std::string _file;
        std::string path() const { return _fullPath; }
        std::string fileName() const { return _file; }
        CFile(const std::string& path, const std::string& file)
        : _fullPath(path) , _file(file) {}

    };

    FileCrawler(const std::string& baseDir, const std::string& extension);
    std::vector<CFile> getAllPaths();
    void crawl(const std::string& cDir = "");

    private:
        std::vector<CFile> _foundFiles;
        std::string _base;
        std::string _extension;


};


#endif //guard