//Based on the work of Andrew Krepps
#include <stdio.h>
#include <math.h>
/*
* Used for multiplying two square matrices of the same size.
* Uses shared memory to store matrix c until it is time to copy
* the final array out to the CPU.
*/
__global__ void dynSharedMatMult(float *A, float *B, const int sideLen) {
    const uint cola = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint rowa = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    const idx = (rowa*sideLen) + cola;
    
    __shared__ float a[sideLen][sideLen];
    __shared__ float b[sideLen][sideLen];

    
    
    // Hold threads in block for copy to finish
    __syncthreads();
    
    
    C[idx] = A[idx] * B[idx];
}

/*
* Used for multiplying two square matrices of the same size.
* Uses global memory for matrix c until it is time to copy
* the final array out to the CPU.
*/
__global__ void dotMult(float *A, float *B, float *C, const int sideLen) {
    const uint cola = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint rowa = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    const idx = (rowa*sideLen) + cola;
    
    // const uint colb = rowa;
    // const uint rowb = cola;
    
    C[idx] = A[idx] * B[idx];
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
    printf("Total number of operations: %d\n", totalThreads);
    
    // 
    float *h_A, *h_B, *h_C;
    float *d_a, *d_b, *d_c;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}
