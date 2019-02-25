//Based on the work of Andrew Krepps
#include <stdio.h>
#include <chrono>


__global__ void saxpy_pinned(int n, float a, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a*x[i] + y[i];
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
	
	printf("Total number of operations: %d\n", totalThreads);
// =======================================================================================
// Start my code

    float *h_a_page, *h_b_page, *h_res_page;
    float *h_a_pin, *h_b_pin, *h_res_pin;

    float *d_a, *d_b;

    // host pageable memory
    h_a_page    = (float*)malloc(totalThreads*sizeof(float));
    h_b_page    = (float*)malloc(totalThreads*sizeof(float));
    h_res_page  = (float*)malloc(totalThreads*sizeof(float));
    // host pinned memory
    cudaMallocHost((void**)&h_a_pin,    totalThreads*sizeof(float));
    cudaMallocHost((void**)&h_b_pin,    totalThreads*sizeof(float));
    cudaMallocHost((void**)&h_res_pin,  totalThreads*sizeof(float));
    // device memory
    cudaMalloc((void**)&d_a, totalThreads*sizeof(float));
    cudaMalloc((void**)&d_b, totalThreads*sizeof(float));

    for(int i=0; i< totalThreads; i++) {
        h_a_page[i] = 1.0f;
        h_a_pin[i]  = 1.0f;
        
        h_b_page[i] = 2.0f;
        h_b_pin[i]  = 2.0f;
    }
    
    // Implement with paged memory
    auto start_paged = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(d_a, h_a_page, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_page, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    
    saxpy_pinned<<<numBlocks, blockSize>>>(totalThreads, 1.0f, d_a, d_b);
    
    cudaMemcpy(h_res_page, d_b, totalThreads*sizeof(float), cudaMemcpyDeviceToHost);
    
    auto stop_paged  = std::chrono::high_resolution_clock::now();
    
    
    // Implement with pinned memory
    auto start_pinned = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(d_a, h_a_pin, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_pin, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    
    saxpy_pinned<<<numBlocks, blockSize>>>(totalThreads, 1.0f, d_a, d_b);
    
    cudaMemcpy(h_res_pin, d_b, totalThreads*sizeof(float), cudaMemcpyDeviceToHost);
    
    auto stop_pinned  = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff_paged = stop_paged-start_paged;
    std::chrono::duration<double> diff_pinned = stop_pinned-start_pinned;
    
    bool equal=true;
    for(int i=0; i<totalThreads; i++) {
        if(h_res_page[i] != h_res_pin[i]) {
            printf("pagedResult[%d] = %f \npinnedResult[%d] = %f\nThese results do not match and they should!", i, h_res_page[i], i, h_res_pin[i]);
            equal=false;
        }
    }
    
    if(equal)
        printf("All results for paged and pinned memory are the same!\n\n");
    
    
    if(diff_paged.count() > diff_pinned.count())
        printf("Pinned memory ran %fs faster than paged, or %fx as fast\n", diff_paged.count()-diff_pinned.count(), diff_paged.count()/diff_pinned.count());
    else
        printf("Paged memory ran %fs faster than pinned, or %fx as fast\n", diff_pinned.count()-diff_paged.count(), diff_pinned.count()/diff_paged.count());
        
    printf("runtime for paged: %f\nruntime for pinned: %f\n\n", diff_paged.count(), diff_pinned.count());
    
    
    
    cudaFree(d_a);
	cudaFree(d_b);
	cudaFreeHost(h_a_pin);
	cudaFreeHost(h_b_pin);
	cudaFreeHost(h_res_pin);
	
	
	
	
}
