/*
 * filterGen.cu
 *
 *  Created on: Apr 22, 2019
 *      Author: zetser
 */

#include "filterGen.h"

__constant__ float k;
__constant__ float pix_DX;
__constant__ float pix_DY;
__constant__ int pixHeight;
__constant__ int pixWidth;

__global__ void calcRelativePosition(float2 *filt, float *zDist) {

    uint row = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint col = (blockIdx.y * blockDim.y) + threadIdx.y;

    uint rel_xPixPos = (col - pixWidth/2)*pix_DX;
	uint rel_yPixPos = (row - pixHeight/2)*pix_DY;

//	filt[row*imgWidth+col].x =
//    filt[row*imgWidth+col].y =

	//int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
	//int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x)+ threadIdx.x;

//	int pixelPos = (row*arraySize)+col;
//
//	rows[pixelPos] = row-arraySize/2;
//	cols[pixelPos] = col-arraySize/2;
	//return threadId;

	printf("xPos: %3.2f yPos: %3.2f \n", rel_xPixPos, rel_yPixPos);
}

void calcSpatImpulseResponse(cimg_library::CImg<unsigned char> img) {

    const int width = img.width();
    const int height = img.height();
    const int numPix = width*height;

    const int filtSize = width*height*sizeof(float2);
    const int zSize = width*height*sizeof(float);

// ============================================================================
    // Let h_zDist be the distance from the object plane to the image plane.
    // Constant for simplicity for now.
    float *h_zDist;
    h_zDist = new float[width*height];
    for(int i=0; i<numPix; i++)
        h_zDist[i] = .001;

    float *d_zDist;     // Z distance specified for each pixel, currently setting as 1 value, but should be variable for future changes.

    cudaMalloc((void**)&d_zDist, zSize);
    cudaMemcpy(d_zDist, h_zDist, zSize, cudaMemcpyHostToDevice);
    // ============================================================================
    // Allocate device memory for filter
    float2 *d_filter;   // filter that will be calculated based on position by GPU
    cudaMalloc((void**)&d_filter, filtSize);
    // ============================================================================
    // Setup Constants
    const float waveLength = 550*pow(10,-6);        // 550nm -> green light
    const float kVal = 2*(M_PI)/waveLength;
    const float imWidth  = 0.001;                   // 1mm
    const float imHeight = 0.001;                   // 1mm
    const float dx = imWidth/(width-1);
    const float dy = imHeight/(height-1);

    cudaMemcpyToSymbol(k,         &kVal,      sizeof(float));
    cudaMemcpyToSymbol(pix_DX,    &dx,        sizeof(float));
    cudaMemcpyToSymbol(pix_DY,    &dy,        sizeof(float));
    cudaMemcpyToSymbol(pixHeight, &height,  sizeof(float));
    cudaMemcpyToSymbol(pixWidth,  &width,   sizeof(float));



    // ============================================================================

    // Generate impulse response of free space propagation
    const int blockSize = 1024;
    const int numBlocks = numPix/blockSize;

    std::cout << "Running kernel now..." << std::endl;
    calcRelativePosition<<<numBlocks, blockSize>>>(d_filter, d_zDist);

    cudaFree(d_zDist);
    //cudaFree(d_filter);
}