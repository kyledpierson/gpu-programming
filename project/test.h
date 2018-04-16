#ifndef __test_h__
#define __test_h__

#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuComplex.h>

#include "scatter.h"
#include "iohandler.h"
#include "JobScheduler.h"



void test_schedule(JobScheduler* scheduler);

#endif 