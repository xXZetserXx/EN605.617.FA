//Based on the work of Andrew Krepps
#include <stdio.h>

#define CONST_SIZE 1024

__constant__ unsigned int constDevA[CONST_SIZE];
__constant__ unsigned int constDevB[CONST_SIZE];

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


__global__ void gpuRegKern_registers(unsigned int *A, unsigned int *B, const int num_elem) {
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(idx < (num_elem)) {
		// Load data from global memory into registers
		unsigned int a = A[idx];
		unsigned int b = B[idx];
		unsigned int c = 0;
		
		// Perform some math on the array values
		c = (a*b) + a + b;
		c = c*c;
		A[idx] = c;
	}
}


__global__ void gpuRegKern_shared(unsigned int *A, unsigned int *B, const int num_elem) {
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(idx < num_elem) {
		extern __shared__ unsigned int concatCoeff[];
		
		concatCoeff[0] = A[idx];
		concatCoeff[1] = B[idx];
		
		concatCoeff[0] = (concatCoeff[0]*concatCoeff[1]) + concatCoeff[0] + concatCoeff[1];
		A[idx] = concatCoeff[0]*concatCoeff[0];
	}
}


__global__ void gpuRegKern_const(unsigned int *C, const int num_elem) {
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(idx < num_elem) {
		C[idx] = (constDevA[idx]*constDevB[idx] + constDevA[idx] + constDevB[idx]) * (constDevA[idx]*constDevB[idx] + constDevA[idx] + constDevB[idx]);
	}
}


__host__ void kernelCaller_Reg(const unsigned int totalThreads, const unsigned int blockSize, const unsigned int numBlocks) {
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
	
	// Timestamp before copying device to device, running kernel, and copy back
	cudaEvent_t startT = get_time();
	
	// Copy data from host arrays to device arrays
	cudaMemcpy(d_a, h_A, num_byte, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_B, num_byte, cudaMemcpyHostToDevice);
	
	// Perform call to kernel
	gpuRegKern_registers <<<numBlocks, blockSize>>>(d_a, d_b, totalThreads);
	
	// Copy resultant array from device memory to host array
	cudaMemcpy(h_C, d_a, num_byte, cudaMemcpyDeviceToHost);
	
	// Timestamp before copying device to device, running kernel, and copy back
	cudaEvent_t stopT = get_time();
	
	//cudaThreadSynchronize();		// Wait for GPU kernels to complete
	cudaEventSynchronize(stopT);		// cudaThreadSynchronize is deprecated, but might need older version to work on vocareum
	
	float delta = 0;
	cudaEventElapsedTime(&delta, startT, stopT);
	
	printf("Elapsed time for performing %d calculations using register memory: %f\n", totalThreads, delta);
	// Print results if you want screen spam
	/*printf("Operation being performed: (a*b)+a+b)^2\n\n");
	for( int i=0; i<totalThreads; i++) {
		printf("Idx: %7d A: %2d B: %2d Result: %d\n", i, h_A[i], h_B[i], h_C[i]);
	}*/
	
	cudaFree((void*) d_a);
	cudaFree((void*) d_b);
	cudaDeviceReset();
}


__host__ void kernelCaller_Shar(const unsigned int totalThreads, const unsigned int blockSize, const unsigned int numBlocks) {
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
	
	// Timestamp before copying device to device, running kernel, and copy back
	cudaEvent_t startT = get_time();
	
	// Copy data from host arrays to device arrays
	cudaMemcpy(d_a, h_A, num_byte, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_B, num_byte, cudaMemcpyHostToDevice);
	
	// Perform call to kernel
	gpuRegKern_shared <<<numBlocks, blockSize,totalThreads*2*sizeof(unsigned int)>>>(d_a, d_b, totalThreads);
	
	// Copy resultant array from device memory to host array
	cudaMemcpy(h_C, d_a, num_byte, cudaMemcpyDeviceToHost);
	
	// Timestamp before copying device to device, running kernel, and copy back
	cudaEvent_t stopT = get_time();
	
	//cudaThreadSynchronize();		// Wait for GPU kernels to complete
	cudaEventSynchronize(stopT);		// cudaThreadSynchronize is deprecated, but might need older version to work on vocareum
	
	float delta = 0;
	cudaEventElapsedTime(&delta, startT, stopT);
	
	printf("Elapsed time for performing %d calculations using shared memory: %f\n", totalThreads, delta);
	
	// Print results if you want screen spam
	/*printf("Operation being performed: (a*b)+a+b)^2\n\n");
	for( int i=0; i<totalThreads; i++) {
		printf("Idx: %7d A: %2d B: %2d Result: %d\n", i, h_A[i], h_B[i], h_C[i]);
	}*/
	
	cudaFree((void*) d_a);
	cudaFree((void*) d_b);
	cudaDeviceReset();
}


__host__ void kernelCaller_const(const unsigned int totalThreads, const unsigned int blockSize, const unsigned int numBlocks) {
	const unsigned int num_byte = CONST_SIZE*sizeof(unsigned int);

	unsigned int *h_A, *h_B, *h_C;
	unsigned *d_c;
	
	// Allocate host memory
	h_A = (unsigned int*)malloc(num_byte);
	h_B = (unsigned int*)malloc(num_byte);
	h_C = (unsigned int*)malloc(num_byte);
	// Allocate device memory
	cudaMalloc((void**)&d_c, num_byte);
	
	// Fill host arrays with random data
	generate_rand_data(h_A, CONST_SIZE);
	generate_rand_data(h_B, CONST_SIZE);
	
	// Timestamp before copying device to device, running kernel, and copy back
	cudaEvent_t startT = get_time();
	
	// Copy random data to constant arrays
	cudaMemcpyToSymbol(constDevA, h_A, num_byte);
	cudaMemcpyToSymbol(constDevB, h_B, num_byte);
	
	gpuRegKern_const<<<CONST_SIZE/256, 256>>>(d_c, CONST_SIZE);
	
	cudaMemcpy(h_C, d_c, num_byte, cudaMemcpyDeviceToHost);
	
	// Timestamp before copying device to device, running kernel, and copy back
	cudaEvent_t stopT = get_time();
	
	//cudaThreadSynchronize();		// Wait for GPU kernels to complete
	cudaEventSynchronize(stopT);		// cudaThreadSynchronize is deprecated, but might need older version to work on vocareum
	
	float delta = 0;
	cudaEventElapsedTime(&delta, startT, stopT);
	
	printf("Elapsed time for performing %d calculations using constant memory: %f\n", CONST_SIZE, delta);
	
	// Print results if you want screen spam
	/*printf("Operation being performed: (a*b)+a+b)^2\n\n");
	for( int i=0; i<CONST_SIZE; i++) {
		printf("Idx: %7d A: %2d B: %2d Result: %d\n", i, h_A[i], h_B[i], h_C[i]);
	}*/
	
	cudaFree(d_c);
	cudaDeviceReset();
	
}


int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 6);
	int blockSize = 16;
	
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
	
	printf("====================================\nRunning with %d elements\n\n", totalThreads);
	
	kernelCaller_Reg(totalThreads, blockSize, numBlocks);
	kernelCaller_Shar(totalThreads, blockSize, numBlocks);
	kernelCaller_const(totalThreads, blockSize, numBlocks);
	
	printf("====================================\n\n\n");
	
	return EXIT_SUCCESS;
}
