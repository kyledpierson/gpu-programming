#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "iohandler.h"
#include "scatter.h"
#include "JobScheduler.h"
#include "FileCrawler.h"
#include "test.h"

#define DEFAULT_FILENAME "uiuc_sample.ppm"

void scheduleForTransformation(JobScheduler* scheduler, const char* inputFile, const std::string& outputFile)
{
    bool fourier = false;
    bool separable = false;

    int x_size, y_size, maxval;
    unsigned int *image = read_ppm((char*)inputFile, x_size, y_size, maxval);

    //std::cout << maxval << std::endl;

    int bytes = x_size * y_size * sizeof(int);
    float *fimage = (float*) mem_check(malloc(bytes));

    float *gaussian, *morlet_1, *morlet_2;
    if (fourier) {
        gaussian = (float*) mem_check(malloc(bytes));
        morlet_1 = (float*) mem_check(malloc(bytes));
        morlet_2 = (float*) mem_check(malloc(bytes));
    }

    // Copy to float image
    for(int i = 0; i < x_size*y_size; i++) {
        fimage[i] = (float) image[i] / maxval;
    }
    free(image);

    // Account for downsampling
    int ds_x_size_1 = x_size>>1;
    int ds_y_size_1 = y_size>>1;
    int ds_bytes_1 = ds_x_size_1 * ds_y_size_1 * sizeof(int);

    int ds_x_size_2 = x_size>>2;
    int ds_y_size_2 = y_size>>2;
    int ds_bytes_2 = ds_x_size_2 * ds_y_size_2 * sizeof(int);

    // Compute the scattering transform
    scatter(fimage, scheduler, outputFile,
            x_size, y_size, bytes,
            ds_x_size_1, ds_y_size_1, ds_bytes_1,
            ds_x_size_2, ds_y_size_2, ds_bytes_2,
            fourier, separable, gaussian, morlet_1, morlet_2);
}

int main(int argc, char **argv) {
    // Read file
    char *filename = strdup(DEFAULT_FILENAME);
    if (argc > 1) {
        filename = strdup(argv[1]);
        fprintf(stderr, "Using %s\n", filename);
    } 

    // Compute the scattering transform
    JobScheduler scheduler(0);
    JobScheduler* sPointer = &scheduler;

    Log::initLogFile("scatter.log");
    LOG_FILE("Starting Scatter log");
    //Pre-process for all
    FileCrawler crawler("source",".ppm");
    crawler.crawl();
    initConsts();
    ThreadPool tp(1);
    for(auto file : crawler.getAllPaths())
    {
    //    LOG_DEBUG(file.path());

        //hack for new destination
        std::string out = std::string("output/") + file.fileName() + ".out";
        //scheduleForTransformation(sPointer,file.path().c_str(),out);
        if(out.size() > 0 && file.path().size() > 0)
        {
            LOG_INFO(file.path() + " -> " + out);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            while(sPointer->currentJobs() > 10)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

            tp.queueJob(([=] () {
                scheduleForTransformation(sPointer,file.path().c_str(),out);
                }));
        }
        else
        {
            LOG_INFO("THATS ODD");
        }
    }
    /*
    scheduleForTransformation(&scheduler, filename, "result.ppm");
    scheduleForTransformation(&scheduler, filename, "result.ppm2");
    scheduleForTransformation(&scheduler, filename, "result.ppm3");
    scheduleForTransformation(&scheduler, filename, "result.ppm4");
    scheduleForTransformation(&scheduler, filename, "result.ppm5");
    */

    LOG_INFO("Waiting to finish jobs...");
    tp.finishAllJobs();
    LOG_INFO("Done scheduling file loads...");
    scheduler.waitUntilDone();
    LOG_FILE("Finish Scatter log");
    Log::closeLog();
}
