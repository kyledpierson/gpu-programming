CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include \

SOURCE=scattering.cu
EXECUTABLE=scattering
FLAGS=

#FOR DEBUGGING
#FLAGS=-g

$(EXECUTABLE): $(SOURCE)
	$(CC) $(FLAGS) $(INCLUDE) $< -o $@ 

clean:
	rm -f analysis.nvvp core* result.ppm scattering scattering.out scattering.err
