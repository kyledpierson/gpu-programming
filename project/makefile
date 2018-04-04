#CC=/usr/local/cuda/bin/nvcc
#INCLUDE=-I/usr/local/cuda/include
CC=/opt/cuda/bin/nvcc
INCLUDE=-I/opt/cuda/include

SOURCE=main.cpp iohandler.cpp scatter.cu JobScheduler.cpp Job.cpp Log.cpp
EXECUTABLE=scatter
FLAGS=-g --expt-extended-lambda

#FOR DEBUGGING
#FLAGS=-g

$(EXECUTABLE): $(SOURCE)
	$(CC) $(FLAGS) $(INCLUDE) $^ -o $@

clean:
	rm -f analysis.nvvp core* result.ppm scatter scatter.out scatter.err
