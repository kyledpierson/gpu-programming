CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include \

SOURCE=main.cpp iohandler.cpp scatter.cu
EXECUTABLE=scatter
FLAGS=

#FOR DEBUGGING
#FLAGS=-g

$(EXECUTABLE): $(SOURCE)
	$(CC) $(FLAGS) $(INCLUDE) $^ -o $@

clean:
	rm -f analysis.nvvp core* result.ppm scatter scatter.out scatter.err
