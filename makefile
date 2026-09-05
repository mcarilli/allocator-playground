a.out: allocators.cu
	nvcc -lcuda allocators.cu

clean:
	rm a.out
