//Based on the work of Andrew Krepps
#include <stdio.h>
#include <math.h>
#include <chrono>
/*
* Used for multiplying two square matrices of the same size.
* Uses shared memory to store matrix c until it is time to copy
* the final array out to the CPU.
*/

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__global__ void dotMultAddShared(float *A, float *B, float *C, const int sideLen) {
    
    const uint cola = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint rowa = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    const uint idx = (rowa*sideLen) + cola;
    
    extern __shared__ float a[];
    
    a[idx]= A[idx];
    
    // Hold threads in block for copy to finish
    __syncthreads();
    //printf("a[%d]: %f b[%d]: %f\n", idx, a[idx], idx, B[idx]);
    C[idx] = a[idx] * B[idx] + a[idx] + B[idx];
}

/*
* Used for multiplying two square matrices of the same size.
* Uses global memory for matrix c until it is time to copy
* the final array out to the CPU.
*/
__global__ void dotMultAdd(float *A, float *B, float *C, const int sideLen) {
    uint cola = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint rowa = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    int idx = (rowa*sideLen) + cola;
    
    C[idx] = A[idx] * B[idx] + A[idx] + B[idx];
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
	
// =======================================================================================
// Start my code

    // Using square matrices for ease.
    // Calculate matrix size and adjust totalThreads if it has no integer sqrt.
    const int side = static_cast<int>(sqrt(totalThreads));
    printf("Matrix size: %d x %d\n", side, side);
    if (totalThreads!=(side*side)) {
    	totalThreads = side*side;
    	numBlocks = totalThreads/blockSize;
    }
    
    printf("Total threads: %d\n", totalThreads);
    printf("Block size: %d\n", blockSize);
    printf("Number of blocks: %d\n", numBlocks);
    
    
    // 
    float *h_A, *h_B, *h_C1, *h_C2;
    float *d_a, *d_b, *d_c;
    
    h_A = (float*)malloc(totalThreads*sizeof(float));
    h_B = (float*)malloc(totalThreads*sizeof(float));
    h_C1 = (float*)malloc(totalThreads*sizeof(float));
    h_C2 = (float*)malloc(totalThreads*sizeof(float));
    
    cudaMalloc((void**)&d_a, totalThreads*sizeof(float));
    cudaMalloc((void**)&d_b, totalThreads*sizeof(float));
    cudaMalloc((void**)&d_c, totalThreads*sizeof(float));
    
    for(int i=0; i<totalThreads; i++) {
    	h_A[i] = 2.0f;
    	h_B[i] = 1.0f;
    	h_C1[i] = 0.0f;
    	h_C2[i] = 0.0f;
    }

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start,0);
    cudaEventCreate(&kernel_stop,0);
    
    
    cudaMemcpy(d_a, h_A, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_C1, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(kernel_start, 0);
    
    dotMultAddShared<<<numBlocks, blockSize, totalThreads*sizeof(int)>>>(d_a, d_b, d_c, side);
    
    cudaEventRecord(kernel_stop, 0);
    cudaEventSynchronize(kernel_stop);
    
    cudaMemcpy(h_C1, d_c, totalThreads*sizeof(float), cudaMemcpyDeviceToHost);
    
    float delta = 0.0f;
    cudaEventElapsedTime(&delta, kernel_start, kernel_stop);
    printf("duration using shared memory: %fmsn\n", delta);

    
    cudaEventRecord(kernel_start, 0);
    
    dotMultAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, side);
    
    cudaEventRecord(kernel_stop, 0);
    cudaEventSynchronize(kernel_stop);
    
    cudaMemcpy(h_C2, d_c, totalThreads*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventElapsedTime(&delta, kernel_start, kernel_stop);
    printf("duration using global memory: %fmsn\n", delta);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    bool different = false;
    for(int i=0; i<totalThreads; i++) {
    	if(h_C1[i] != h_C2[i])
    		different = true;
    }
    
    if(different)
    	printf("Different values detected at [%d]: h_C1=%f h_C2=%f\n", 0, h_C1[0], h_C2[0]);
    else
    	printf("%2.1f*%2.1f + %2.1f + %2.1f = %2.1f\n", h_A[0], h_B[0], h_A[0], h_B[0], h_C1[0]);
    
}
