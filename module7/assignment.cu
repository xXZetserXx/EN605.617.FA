#include <stdio.h>
#include <time.h>
#include <cuda.h>

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

// Pulled from module 6 assignment
__host__ void generate_rand_data(int * host_data_ptr, const int num_elem)
{
	// Generate random values from 0-19
    for(unsigned int i=0; i < num_elem; i++)
    	host_data_ptr[i] = (int) (rand()%20);
}

__global__ void gpuKernel(int *A, int *B, int *C, const int num_elem) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < (num_elem)) {
		// Load data from global memory into registers
		int a = A[idx];
		int b = B[idx];
		int c = 0;
		
		// Perform some math on the array values
		c = (a*b) + a + b;
		c = c*c;
		C[idx] = c;
	}
}


__host__ float execute_concurrent_streamed_kernel(int arraySize, int N, int tpb) {    
    const int h_byteSize = arraySize*sizeof(int);
    const int d_byteSize = N*sizeof(int);
    
    
    cudaDeviceProp prop;
    int whichDevice;
    
    // Following taken from the book "CUDA by Example"
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    // "A GPU supporting device overlap possesses the capacity to simultaneously
    // execute a CUDA C kernel while performing a copy between device and host memory."
    if(!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no speedup from streams\n");
        return 0;
    }
    

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    
    int *h_a, *h_b, *h_c;
    // Create two sets of GPU buffers
    int *d_a0, *d_b0, *d_c0;    // Buffers used in stream0
    int *d_a1, *d_b1, *d_c1;    // Buffers used in stream1
    
    // Allocate pinned memory, cudaMemcpyAsync requires host memory be page-locked
    cudaHostAlloc((void **)&h_a, h_byteSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_b, h_byteSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_c, h_byteSize, cudaHostAllocDefault);
    
    // Used in stream0
    cudaMalloc((void**) &d_a0, d_byteSize);
    cudaMalloc((void**) &d_b0, d_byteSize);
    cudaMalloc((void**) &d_c0, d_byteSize);
    // Used in stream1
    cudaMalloc((void**) &d_a1, d_byteSize);
    cudaMalloc((void**) &d_b1, d_byteSize);
    cudaMalloc((void**) &d_c1, d_byteSize);
    
    // Fill host arrays with random data
	generate_rand_data(h_a, arraySize);
	generate_rand_data(h_b, arraySize);
	
	// Timers
    cudaEvent_t startT, stopT;
    float deltaT;
    
    startT = get_time();
    
    /* =================================================================================
     * We are only copying part of full data each time. Queueing in a ping-pong fashion
     * like shown below optimizes the execution timeline. Trying to queue all stream0
     * operations and then queue all stream1 operations will cause the copy back to host
     * memory in stream0 to block the copy to device for stream1. Now copies can start
     * in stream1 while stream0's kernel is executing.
     * =================================================================================
     */
	for(int i=0; i<arraySize; i+=N*2) {
        // Queue up copy of data for a array in both streams
        cudaMemcpyAsync(d_a0, h_a+i,    d_byteSize, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_a1, h_a+i+N,  d_byteSize, cudaMemcpyHostToDevice, stream1);
        // Queue up copy of data for b array in both streams
        cudaMemcpyAsync(d_b0, h_b+i,    d_byteSize, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_b1, h_b+i+N,  d_byteSize, cudaMemcpyHostToDevice, stream1);
        // Queue up running of gpu kernel
        gpuKernel<<<N/tpb, tpb, 0, stream0>>>(d_a0, d_b0, d_c0, N);
        gpuKernel<<<N/tpb, tpb, 0, stream1>>>(d_a1, d_b1, d_c1, N);
        // Queue up copy of data from device to pinned memory
        cudaMemcpyAsync(h_c+i,      d_c0, d_byteSize, cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_c+i+N,    d_c1, d_byteSize, cudaMemcpyDeviceToHost, stream1);
	    
	}
	
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	
	stopT = get_time();
	cudaEventSynchronize(stopT);
	
	cudaEventElapsedTime(&deltaT, startT, stopT);
	
	
	// Cleanup memory
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaFree(d_a0);
	cudaFree(d_b0);
	cudaFree(d_c0);
	cudaFree(d_a1);
	cudaFree(d_b1);
	cudaFree(d_c1);
	cudaEventDestroy(startT);
	cudaEventDestroy(stopT);
	
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	return deltaT;
}

__host__ float execute_kernel(int arraySize, int tpb) {    
    const int h_byteSize = arraySize*sizeof(int);    
    
    
    int *h_a, *h_b, *h_c;
    // Create two sets of GPU buffers
    int *d_a, *d_b, *d_c;    // Buffers used in stream1
    
    // Allocate pinned memory, cudaMemcpyAsync requires host memory be page-locked
    cudaHostAlloc((void **)&h_a, h_byteSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_b, h_byteSize, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_c, h_byteSize, cudaHostAllocDefault);
    
    cudaMalloc((void**) &d_a, h_byteSize);
    cudaMalloc((void**) &d_b, h_byteSize);
    cudaMalloc((void**) &d_c, h_byteSize);
    
    // Fill host arrays with random data
	generate_rand_data(h_a, arraySize);
	generate_rand_data(h_b, arraySize);
	
	// Timers
    cudaEvent_t startT, stopT;
    float deltaT;
    
    startT = get_time();
    
	// Copy data to device
    cudaMemcpy(d_a, h_a, h_byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, h_byteSize, cudaMemcpyHostToDevice);
    
    gpuKernel<<<arraySize/tpb, tpb>>>(d_a, d_b, d_c, arraySize);
	
	cudaMemcpy(h_c, d_c, h_byteSize, cudaMemcpyDeviceToHost);
	
	stopT = get_time();
	cudaEventSynchronize(stopT);
	
	cudaEventElapsedTime(&deltaT, startT, stopT);
	
	
	// Cleanup memory
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaEventDestroy(startT);
	cudaEventDestroy(stopT);

	return deltaT;
}


int main(int argc, char **argv) {
    int arraySize = 4096;     // Total number of elements to process.
    int N = 256;              // Number of elements to pass to GPU at one time.
    int tpb = 128;
    
    if(argc >= 2)
    	arraySize = atoi(argv[1]);
	if(argc >= 3)
		N = atoi(argv[2]);
	if(argc >= 4)
		tpb = atoi(argv[3]);
    
    if(arraySize % N !=0 || N%tpb!=0) {
    	printf("Number of total threads is not divisible by number of elements to process in each stream iteration.\n");
    	return 0;
    }

	float delta_concurrent = execute_concurrent_streamed_kernel(arraySize, N, tpb);
	float delta_normal = execute_kernel(arraySize, tpb);

	printf("========================\n");
	printf("Summary\n");
	printf("Total Threads: %d\n", arraySize);
	printf("Number of concurrent kernel instances: %d\n", N);
	printf("Thread Size: %d\n", tpb);
	printf("========================\n");
	printf("Time to copy memory and execute kernel with two streams running concurrently.\n");
	printf("duration: %fms\n",delta_concurrent);
	printf("========================\n");
	printf("Time to copy memory and execute kernel running using a normal kernel execution.\n");
	printf("duration: %fms\n\n\n",delta_normal);
    
}


