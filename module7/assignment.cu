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
		unsigned int a = A[idx];
		unsigned int b = B[idx];
		unsigned int c = 0;
		
		// Perform some math on the array values
		c = (a*b) + a + b;
		c = c*c;
		C[idx] = c;
	}
}


int main(int argc, char **argv) {
    const int arraySize = 4096;     // Total number of elements to process.
    const int N = 128;              // Number of elements to pass to GPU at one time.
    
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
    
    // Timers
    cudaEvent_t startT, stopT;
    float *deltaT;
    
    startT = get_time();
    
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
	
    /* =================================================================================
     * For each stream queue up copy of data from host pinned memory to device memory.
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
        gpuKernel<<<N/128, 128, 0, stream0>>>(d_a0, d_b0, d_c0, N);
        gpuKernel<<<N/128, 128, 0, stream1>>>(d_a1, d_b1, d_c1, N);
        // Queue up copy of data from device to pinned memory
        cudaMemcpyAsync(h_c+i,      d_c0, d_byteSize, cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_c+i+N,    d_c1, d_byteSize, cudaMemcpyDeviceToHost, stream1);
	    
	}
	
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	
	stopT = get_time();
	cudaEventSynchronize(stopT);
	
	//cudaEventElapsedTime(deltaT, startT, stopT);
	//printf("Elasped Time: %fms\n", *deltaT);
	
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
	
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
    
    
}
