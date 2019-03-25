#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <stdint.h>
#include <math.h>
#include <cufft.h>

typedef float2 Complex;


const uint16_t imgSize = 128;

/*
 * Kernel to calculate relative position from center for a given pixel in a vectorized 2D image/array
 */
__global__ void calcRelativePosition(float *pos, int arraySize)
{
	uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint col = idx%arraySize;
    uint row = idx/arraySize;

	//int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x)+ threadIdx.x;

	int pixelPos = (row*arraySize)+col;

	uint rel_row = row-arraySize/2;
	uint rel_col = col-arraySize/2;

	float relative = sqrt((float)((rel_col*rel_col)+ (rel_row*rel_row)));
	pos[pixelPos] = relative;

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


__host__ void loadImage() {
	// Just loading array of random values.

}


__host__ void driver() {
	const int totalPixels = imgSize*imgSize;
	const int blockSize = 256;
	const int numBlocks = totalPixels/blockSize;

	const int size_bytes = totalPixels*sizeof(float);


	float *h_relPos;
	float *d_relPos;

	h_relPos    = (float*)malloc(size_bytes);
	cudaMalloc((void**)&d_relPos, size_bytes);

	calcRelativePosition<<<numBlocks,blockSize>>>(d_relPos, imgSize);

	cudaMemcpy(h_relPos, d_relPos, size_bytes, cudaMemcpyDeviceToHost);

    int center = (imgSize/2);
    
    printf("Relative Position: %3.3f for (%d,%d)", h_relPos[center*imgSize+(center+20)], center+20, center);


	cudaFree(d_relPos);
}



int main(int argc, char** argv)
{
    driver();
	return 0;
}
