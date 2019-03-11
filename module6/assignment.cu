//Based on the work of Andrew Krepps
#include <stdio.h>

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

// Following function pulled from register.cu class file
__host__ void generate_rand_data(unsigned int * host_data_ptr, const unsigned int num_elem)
{
        for(unsigned int i=0; i < num_elem; i++)
        {
                host_data_ptr[i] = (unsigned int) (rand()%20);
        }
}


__global__ void gpuRegKern(unsigned int *A, unsigned int *B, const int num_elem) {
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(idx < (num_elem)) {
		// Load data from global memory into registers
		float a = A[idx];
		float b = B[idx];
		float c = 0;
		
		// Perform some math on the array values
		c = (a*b) + a + b;
		c = c*c;
		A[idx] = c;
	}
}


__host__ void kernelCaller(const unsigned int totalThreads, const unsigned int blockSize, const unsigned int numBlocks) {
	const unsigned int num_byte = totalThreads*sizeof(unsigned int);	// Calculate memory size of arrays
	
	unsigned int *h_A, *h_B, *h_C;
	unsigned int *d_a, *d_b;
	
	// Allocate host memory
	h_A = (unsigned int*)malloc(num_byte);
	h_B = (unsigned int*)malloc(num_byte);
	h_C = (unsigned int*)malloc(num_byte);
	// Allocate device memory
	cudaMalloc((void**)&d_a, num_byte);
	cudaMalloc((void**)&d_b, num_byte);
	
	// Fill host arrays with random data
	generate_rand_data(h_A, totalThreads);
	generate_rand_data(h_B, totalThreads);
	
	// Copy data from host arrays to device arrays
	cudaMemcpy(d_a, h_A, num_byte, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_B, num_byte, cudaMemcpyHostToDevice);
	
	// Perform call to kernel
	gpuRegKern <<<numBlocks, blockSize>>>(d_a, d_b, totalThreads);
	
	//cudaThreadSynchronize();		// Wait for GPU kernels to complete
	cudaDeviceSynchronize();		// cudaThreadSynchronize is deprecated, but might need older version to work on vocareuml
	cudaGetLastError();
	
	// Copy resultant array from device memory to host array
	cudaMemcpy(h_C, d_a, num_byte, cudaMemcpyDeviceToHost);
	
	printf("Operation being performed: (a*b)+a+b)^2\n\n");
	for( int i=0; i<totalThreads; i++) {
		printf("Idx: %7d A: %2d B: %2d Result: %d\n", i, h_A[i], h_B[i], h_C[i]);
	}
	
	cudaFree((void*) d_a);
	cudaFree((void*) d_b);
	cudaDeviceReset();
}


int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	
	kernelCaller(totalThreads, blockSize, numBlocks);
	
	return EXIT_SUCCESS;
}
