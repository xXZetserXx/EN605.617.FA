#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_functions.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <stdint.h>
#include <math.h>
#include <cufft.h>
#include <curand.h>
#include <curand_kernel.h>

typedef float2 Complex;


uint16_t imgSize = 128;
uint32_t totalPixels = imgSize*imgSize;
int blockSize = 256;
int numBlocks = totalPixels/blockSize;


__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

/*
 * Kernel to calculate relative position from center
 * in pixel space for a given pixel in a vectorized 2D image/array.
 */
__global__ void calcRelativePosition(float *pos, int arraySize)
{
	uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = idx%arraySize;
    int row = idx/arraySize;

	int rel_row = row-arraySize/2;
	int rel_col = col-arraySize/2;

	float relative = sqrt((float)((rel_col*rel_col)+ (rel_row*rel_row)));
	pos[idx] = relative;

	//printf("xPos: %3u yPos: %3u \nPixel Distance from Center: %3.3f\n", col, row, relative);

}

/*
 * Kernel to multiply two complex matrices
 */
__global__ void ComplexMUL(Complex *a, Complex *b)
{
	// Adapted from cufft_example.cu file.
	int i = threadIdx.x + blockIdx.x*blockDim.x;
    a[i].x = a[i].x * b[i].x - a[i].y*b[i].y;
    a[i].y = a[i].x * b[i].y + a[i].y*b[i].x;
}


/*
 * Kernel to generate random seeds, taken from curand_example.cu
 */
__global__ void init(unsigned int seed, curandState_t* states) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
				idx, /* the sequence number should be different for each core (unless you want all
							 cores to get the same sequence of numbers for some reason - use thread id! */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&states[idx]);
}


/*
 * Kernel to generate random numbers, taken from curand_example.cu
 */
__global__ void randoms(curandState_t* states, unsigned int* numbers) {
	/* curand works like rand - except that it takes a state as a parameter */
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	numbers[idx] = curand(&states[idx]) % 256;
}


/*
 * Host function which will read in an input image.
 * Just using cuRAND to generate random data for now.
 * Code for using cuRand was adapted from curand_example.cu
 */
__host__ unsigned int* loadImage() {
	// Generate random states
	curandState_t* states;
	cudaMalloc((void**) &states, totalPixels*sizeof(curandState_t));
	init<<<numBlocks, blockSize>>>(time(0), states);

	// Generate random numbers from random states
	unsigned int* h_randImg = new unsigned int[totalPixels];
	unsigned int* d_randImg;


	cudaEvent_t startT, stopT;
	float deltaT;

	startT = get_time();

	cudaMalloc((void**)&d_randImg, totalPixels*sizeof(unsigned int));	// Size is already in bytes because we're using uint8_t, but w/e

	randoms<<<numBlocks, blockSize>>>(states, d_randImg);

	cudaMemcpy(h_randImg, d_randImg, totalPixels*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	stopT = get_time();
	cudaEventSynchronize(stopT);

	cudaEventElapsedTime(&deltaT, startT, stopT);

	printf("Generated random image in: %fms\n", deltaT);

//	for(int i=0; i<imgSize; i++) {
//		for(int j=0; j<imgSize; j++) {
//			printf("%d ", h_randImg[i*imgSize+j]);
//		}
//		printf("\n");
//	}

	cudaFree(states);
	cudaFree(d_randImg);

	return h_randImg;
}


__host__ float* genRelativePos() {
	const int size_bytes = totalPixels*sizeof(float);


	float *h_relPos;
	float *d_relPos;

	cudaEvent_t startT, stopT;
	float deltaT;

	startT = get_time();
	// Begin kernel execution
	h_relPos    = (float*)malloc(size_bytes);
	cudaMalloc((void**)&d_relPos, size_bytes);
	// Generate array of positions relative to center pixel (in pixel space)
	calcRelativePosition<<<numBlocks,blockSize>>>(d_relPos, imgSize);
	// Copy data back to CPU
	cudaMemcpy(h_relPos, d_relPos, size_bytes, cudaMemcpyDeviceToHost);

	stopT = get_time();
	cudaEventSynchronize(stopT);

	cudaEventElapsedTime(&deltaT, startT, stopT);

	printf("Generated filter array in: %fms\n", deltaT);

//    for(int i=0; i<imgSize; i++) {
//    	for(int j=0; j<imgSize; j++) {
//    		printf("%3.3f ", h_relPos[i*imgSize+j]);
//    	}
//    	printf("\n");
//    }

    cudaFree(d_relPos);
    return h_relPos;
}


/*
 * Host function to make use of cuFFT and cuda multiplication to perform filter.
 */
__host__ void performFilter(unsigned int* inputImg, float* filterArray) {
	// Copy input images into real portion of complex signals and zero out imaginary portion
	Complex* h_compImg  = new Complex[totalPixels];
	Complex* h_compFilt = new Complex[totalPixels];
	for(int i=0; i<totalPixels; i++){
		h_compImg[i].x  = (float)inputImg[i];
		h_compImg[i].y = 0;
		h_compFilt[i].x = filterArray[i]/1000;	// Let's say 1 pixel = 1mm
		h_compFilt[i].y = 0;
	}
	
	cudaEvent_t startT, stopT;
	float deltaT;

	startT = get_time();
	int mem_size = sizeof(cufftComplex)* totalPixels;
	// Create device signal to filter from input image.
	cufftComplex *d_signal;
	checkCudaErrors(cudaMalloc((void **) &d_signal, mem_size)); 
	checkCudaErrors(cudaMemcpy(d_signal, h_compImg, mem_size, cudaMemcpyHostToDevice));

	// Create device filter signal from filter array input
	cufftComplex *d_filter_kernel;
	checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));
	checkCudaErrors(cudaMemcpy(d_filter_kernel, h_compFilt, mem_size, cudaMemcpyHostToDevice));

	// CUFFT plan
	cufftHandle plan;
	cufftPlan2d(&plan, imgSize, imgSize, CUFFT_C2C);

	// Transform signal and filter
	printf("Transforming signal cufftExecR2C\n");
	cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);
	cufftExecC2C(plan, (cufftComplex *)d_filter_kernel, (cufftComplex *)d_filter_kernel, CUFFT_FORWARD);

	printf("Launching Complex multiplication...\n");
	ComplexMUL<<<numBlocks, blockSize>>>(d_signal, d_filter_kernel);

	// Perform IFFT
	printf("Starting IFFT\n");
	cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

	Complex* result = new Complex[totalPixels];
	cudaMemcpy(result, d_signal, sizeof(Complex)*totalPixels, cudaMemcpyDeviceToHost);


	stopT = get_time();
	cudaEventSynchronize(stopT);
	cudaEventElapsedTime(&deltaT, startT, stopT);

	printf("Performed filter of %dx%d image in: %fms\n\n", imgSize, imgSize, deltaT);

//	for(int i=0; i<imgSize; i++) {
//		for(int j=0; j<imgSize; j++) {
//			printf("%3.3f + %3.3fj  ", result[i*imgSize+j].x, result[i*imgSize+j].y);
//		}
//		printf("\n");
//	}
}


int main(int argc, char** argv)
{
	// This will be our filter
	float* relPos = genRelativePos();
	// This will be our input
	unsigned int* inpImg = loadImage();

	performFilter(inpImg, relPos);

	
	imgSize = 512;
	totalPixels = imgSize*imgSize;
	blockSize = 256;
	numBlocks = totalPixels/blockSize;

	// This will be our filter
	relPos = genRelativePos();
	// This will be our input
	inpImg = loadImage();

	performFilter(inpImg, relPos);

	return 0;
}
