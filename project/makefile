CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include
#CC=/opt/cuda/bin/nvcc
#INCLUDE=-I/opt/cuda/include

SOURCE=main.cpp iohandler.cpp scatter.cu JobScheduler.cpp Job.cpp Log.cpp test.cu ThreadPool.cpp FileCrawler.cpp MemoryWrapper.cu
HEADERS=iohandler.h Job.h JobScheduler.h Log.h ThreadPool.h FileCrawler.h MemoryWrapper.h
EXECUTABLE=scatter
FLAGS=-g --expt-extended-lambda

#FOR DEBUGGING
#FLAGS=-g

$(EXECUTABLE): $(SOURCE) $(HEADERS)
	$(CC) $(FLAGS) $(INCLUDE) $(SOURCE) -o $@ -lcufft

clean:
	rm -f analysis.nvvp core* result*.ppm scatter scatter.out scatter.err
