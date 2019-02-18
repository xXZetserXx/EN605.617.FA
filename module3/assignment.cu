//Based on the work of Andrew Krepps
#include <stdio.h>
#include <chrono>


__global__ void calcPixelPosition(int * d)
{
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	//uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x)+ threadIdx.x;
	
	d[threadId] = i;
	//return threadId;
	
	printf("xPos: %3u yPos: %3u ", i, j);
}


int main(int argc, char** argv)
{
	// read command line arguments
	
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
	    // This is the same as the total image size for this 
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
	
	int cpu_d[totalThreads] = {0};
	int *dev_d;
	cudaMalloc((void**)&dev_d, totalThreads*sizeof(int));
	cudaMemset((void**)&dev_d, 0, totalThreads*sizeof(int));
	
	
	auto start = std::chrono::high_resolution_clock::now();
	
	calcPixelPosition<<<numBlocks,blockSize>>>(dev_d);
	
	auto stop = std::chrono::high_resolution_clock::now();
	
	cudaMemcpy(cpu_d, dev_d, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_d);
	
    cudaDeviceSynchronize();
}
