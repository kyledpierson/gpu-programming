CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include \

SOURCE=sobel.cu
EXECUTABLE=sobel
FLAGS=

#FOR DEBUGGING
#FLAGS=-g

$(EXECUTABLE): $(SOURCE)
	$(CC) $(FLAGS) $(INCLUDE) $< -o $@ 

clean:
	rm -f result*.ppm sobel sobel.out sobel.err analysis.nvvp naive.ppm shared.ppm shared_overlap.ppm unrolled.ppm
