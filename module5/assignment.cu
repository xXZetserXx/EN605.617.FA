//Based on the work of Andrew Krepps
#include <stdio.h>
#include <math.h>
#include <chrono>
/*
* Used for multiplying two square matrices of the same size.
* Uses shared memory to store matrix c until it is time to copy
* the final array out to the CPU.
*/
__global__ void dotMultAddShared(float *A, float *B, float *C, const int sideLen) {
    const uint cola = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint rowa = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    const uint idx = (rowa*sideLen) + cola;
    
    extern __shared__ float a[];
    
    a[idx]= A[idx];
    
    // Hold threads in block for copy to finish
    __syncthreads();
    
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
    printf("Matrix size: %d x %d", side, side);
    totalThreads = side*side;
    printf("Total number of threads: %d\n", totalThreads);
    numBlocks = totalThreads/blockSize;
    
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
    	//printf("h_A[%d]: %f h_B[%d]: %f\n", i, h_A[i], i, h_B[i]);
    	h_C1[i] = 0.0f;
    	h_C2[i] = 0.0f;
    }

    	
    cudaMemcpy(d_a, h_A, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_C1, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    
    auto start_shared = std::chrono::high_resolution_clock::now();
    
    dotMultAddShared<<<numBlocks, blockSize, totalThreads*sizeof(int)>>>(d_a, d_b, d_c, side);
    
    auto stop_shared  = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_C1, d_c, totalThreads*sizeof(float), cudaMemcpyDeviceToHost);
    
    
    auto start_glob = std::chrono::high_resolution_clock::now();
    
    dotMultAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, side);
    
    auto stop_glob  = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_C2, d_c, totalThreads*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    std::chrono::duration<double> diff_shared 	= stop_shared - start_shared;
    std::chrono::duration<double> diff_glob	= stop_glob - start_glob;
    
    bool different = false;
    for(int i=0; i<totalThreads; i++) {
    	if(h_C1[i] != h_C2[i])
    		different = true;
    }
    
    if(different)
    	printf("Different values detected at [%d]: h_C1=%f h_C2=%f\n", 0, h_C1[0], h_C2[0]);
    else
    	printf("%2.1f*%2.1f + %2.1f + %2.1f = %2.1f\n", h_A[0], h_B[0], h_A[0], h_B[0], h_C1[0]);
    
    
    printf("Time elapsed using shared memory: %f\n", diff_shared.count());
    printf("Time elapsed using global memory: %f\n\n", diff_glob.count());
    
}
