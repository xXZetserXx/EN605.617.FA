//Based on the work of Andrew Krepps
#include <stdio.h>
#include <chrono>

__global__ void calcRelativePosition(int arraySize, int *rows, int *cols)
{
	uint col = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint row = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	//int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
	//int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x)+ threadIdx.x;
	
	int pixelPos = (row*arraySize)+col;
	
	rows[pixelPos] = row-arraySize/2;
	cols[pixelPos] = col-arraySize/2;
	//return threadId;
	
	//printf("xPos: %3u yPos: %3u \n", i, j);
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
	
	// =======================================================================================
	// Start my code
	int totalThreadsSq = totalThreads*totalThreads;
	printf("numBlocks: %d blockSize: %d\n", numBlocks, blockSize);
	const dim3 blocks(numBlocks, numBlocks);
	const dim3 threads(blockSize, blockSize);
	
	int cpu_r[totalThreadsSq] = {0};
	int cpu_c[totalThreadsSq] = {0};
	int *dev_r, *dev_c;
	
	cudaMalloc((void**)&dev_r, totalThreadsSq*sizeof(int));
	cudaMalloc((void**)&dev_c, totalThreadsSq*sizeof(int));
	cudaMemset((void**)&dev_r, 0, totalThreadsSq*sizeof(int));
	cudaMemset((void**)&dev_c, 0, totalThreadsSq*sizeof(int));
	
	
	auto start_gpu = std::chrono::high_resolution_clock::now();

	calcRelativePosition<<<blocks,threads>>>(totalThreads, dev_r, dev_c);
	
	auto stop_gpu  = std::chrono::high_resolution_clock::now();
	
	cudaMemcpy(cpu_r, dev_r, totalThreadsSq*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_c, dev_c, totalThreadsSq*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_r);
	cudaFree(dev_c);
	
	/*for(int i=0; i<totalThreadsSq; i++) {
		printf("Position from center: (%d,%d)\n", cpu_r[i], cpu_c[i]);
	}*/
	std::chrono::duration<double> diff_gpu = stop_gpu-start_gpu;
	printf("Time to find relative positions of %dx%d array using GPU: %f s\n",
				 totalThreads, totalThreads, diff_gpu.count());
				 
	cudaDeviceSynchronize();
	// ===========================================================================
	// Now repeat using CPU only		 
	int r[totalThreadsSq] = {0};
	int c[totalThreadsSq] = {0};
	
	auto start_cpu = std::chrono::high_resolution_clock::now();
	
	for(int i=0; i<totalThreads; i++) {
		for(int j=0; j<totalThreadsSq; j++) {
			r[i*totalThreads+j] = i-totalThreads/2;
			c[i*totalThreads+j] = j-totalThreads/2;
		}
	}
	auto stop_cpu  = std::chrono::high_resolution_clock::now();
	
	/*for(int i=0; i<totalThreadsSq; i++) {
		printf("Position from center: (%d,%d)\n", r[i], c[i]);
	}*/
	std::chrono::duration<double> diff_cpu = stop_cpu-start_cpu;
	printf("Time to find relative positions of %dx%d array using CPU: %f s\n",
				 totalThreads, totalThreads, diff_cpu.count());
	
    
}
