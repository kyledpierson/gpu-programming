CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include
#CC=/opt/cuda/bin/nvcc
#INCLUDE=-I/opt/cuda/include

SOURCE=main.cpp iohandler.cpp scatter.cu JobScheduler.cpp Job.cpp Log.cpp test.cu ThreadPool.cpp FileCrawler.cpp
HEADERS=iohandler.h Job.h JobScheduler.h Log.h ThreadPool.h FileCrawler.h
EXECUTABLE=scatter
FLAGS=-g --expt-extended-lambda

#FOR DEBUGGING
#FLAGS=-g

$(EXECUTABLE): $(SOURCE) $(HEADERS)
	$(CC) $(FLAGS) $(INCLUDE) $(SOURCE) -o $@ -lcufft

clean:
	rm -f analysis.nvvp core* output/* result*.ppm scatter scatter.log scatter.out scatter.err
