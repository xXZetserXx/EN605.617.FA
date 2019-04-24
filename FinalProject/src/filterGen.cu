/*
 * filterGen.cu
 *
 *  Created on: Apr 22, 2019
 *      Author: zetser
 */

#include "filterGen.h"

__constant__ float  k;
__constant__ float  pix_DX;
__constant__ float  pix_DY;
__constant__ int    pixWidth;
__constant__ int    pixHeight;
__constant__ float  const_zDist;

__global__ void genFilter(complex *filt) {

    // ===============================================================
    // This part will need to change if I change block/thread paradigm
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint row = idx/pixWidth;
    uint col = idx%pixWidth;
    // ===============================================================

    float x = ((float)col - pixWidth/2.0)  * pix_DX;
	float y = ((float)row - pixHeight/2.0) * pix_DY;

	// Want to store this in global memory and use it for the filter in another kernel.
    filt[idx].x =  ( k/(2*(M_PI)*const_zDist) ) * sin( (k/(2*const_zDist)) * ((x*x)+(y*y)+(const_zDist*const_zDist))  );
    filt[idx].y = -( k/(2*(M_PI)*const_zDist) ) * cos( (k/(2*const_zDist)) * ((x*x)+(y*y)+(const_zDist*const_zDist))  );

//    if(row==256 && col==256) {
//        printf("(%d,%d):\t%3.5f + j%3.5f\n", row, col, filt[idx].x, filt[idx].y);
//    }
}

__global__ void ComplexMUL(complex *a, complex *b)
{
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    float cx = a[i].x * b[i].x - a[i].y * b[i].y;
    float cy = a[i].x * b[i].y + a[i].y * b[i].x;

    a[i].x = cx;
    a[i].y = cy;

//    printf("%f + j%f\n", a[i].x, a[i].y);
}

__global__ void fftshift(complex *in_img, complex *out_img) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint oldRow = idx/pixWidth;
    uint oldCol = idx%pixWidth;

    uint newRow = 0;
    uint newCol = 0;

    if(oldRow<pixHeight && oldCol<pixWidth) {
        if( oldRow<(pixHeight/2-1) ) {
            newRow = oldRow + (pixHeight/2) - 1;
        }
        else {
            newRow = oldRow - (pixHeight/2) - 1;
        }

        if( oldCol<(pixWidth/2-1) ) {
            newCol = oldCol + (pixWidth/2) - 1;
        }
        else {
            newCol = oldCol - (pixWidth/2) - 1;
        }

        out_img[newRow*pixWidth+newCol] = in_img[oldRow*pixWidth+oldCol];


    }

}



__global__ void createHolo(complex* fzp, float* out) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

    out[i] = ( (1+fzp[i].x)*(1+fzp[i].x) ) - ( fzp[i].y * fzp[i].y );
//    fzp[i].x += 1;
//    out[i] = cuCabsf(fzp[i]);
//    printf("%f + j%f\n", fzp[i].x,fzp[i].y);
}

void calcSpatImpulseResponse(cimg_library::CImg<unsigned char> img, float distance) {

//    cimg_library::CImgDisplay main_disp(img,"Original");
//
//    while (!main_disp.is_closed()) {
//        main_disp.wait();
//    }

    const int width = img.width();
    const int height = img.height();
    const int numPix = width*height;
    const float waveLength = 550*pow(10,-6);        // 550nm -> green light
    const float kVal = 2*(M_PI)/waveLength;
    const float imWidth  = 0.001;                   // 1mm
    const float imHeight = 0.001;                   // 1mm
    const float dx = imWidth/(width-1);
    const float dy = imHeight/(height-1);


// ============================================================================
    // Let h_zDist be the distance from the object plane to the image plane.
    // Constant for simplicity for now.
    float *h_zDist;

    const int zSize = sizeof(float);
    h_zDist = (float*)malloc(zSize);
    h_zDist[0] = distance;
    /* For when each pixel may have a different distance
    const int zSize = width*height*sizeof(float);
    h_zDist = new float[width*height];
    for(int i=0; i<numPix; i++)
        h_zDist[i] = .001;

    float *d_zDist;     // Z distance specified for each pixel, currently setting as 1 value, but should be variable for future changes.
    cudaMalloc((void**)&d_zDist, zSize);
    cudaMemcpy(d_zDist, h_zDist, zSize, cudaMemcpyHostToDevice);
    */
// ============================================================================
// Allocate device memory
    const int compImgSize = numPix*sizeof(cufftComplex);

    // Allocate and copy over input Image

    complex *h_image = new complex[numPix];
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++) {
            h_image[(i*width+j)].x = (img(j, i, 0)/255.0);     // Only performing filter on one color channel
            h_image[(i*width+j)].y = 0;
//            printf("Image value: %f\n", h_image[(i*width+j)].x);
        }
    }
    cufftComplex *d_image;
    cudaMalloc((void**)&d_image, compImgSize);
    cudaMemcpy(d_image, h_image, compImgSize, cudaMemcpyHostToDevice);

    std::cout << "Did I make it?" << std::endl;
    // We will be generating the filter based on position and will never pull it out of GPU memory

    cufftComplex *d_filter;   // filter that will be calculated based on position by GPU
    cudaMalloc((void**)&d_filter, compImgSize);
// ============================================================================
// Copy values to constant memory
    cudaMemcpyToSymbol(k,           &kVal,    sizeof(float));
    cudaMemcpyToSymbol(pix_DX,      &dx,      sizeof(float));
    cudaMemcpyToSymbol(pix_DY,      &dy,      sizeof(float));
    cudaMemcpyToSymbol(pixHeight,   &height,  sizeof(float));
    cudaMemcpyToSymbol(pixWidth,    &width,   sizeof(float));
    cudaMemcpyToSymbol(const_zDist, h_zDist,  sizeof(float));
// ============================================================================
// Generate impulse response of free space propagation
    const int blockSize = 1024;
    const int numBlocks = numPix/blockSize;

    std::cout << "Creating filter" << std::endl;
    genFilter <<<numBlocks, blockSize>>>(d_filter);     // Leave filter in GPU global memory for use in cuFFT

// ============================================================================
// Perform FFT on filter and input image
    cufftHandle plan;
    cufftPlan2d(&plan, width, height, CUFFT_C2C);

    // Perform FFTs
    cufftExecC2C(plan, (cufftComplex *)d_filter,     (cufftComplex *)d_filter,     CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *)d_image,      (cufftComplex *)d_image,      CUFFT_FORWARD);

    ComplexMUL <<<numBlocks, blockSize>>>(d_image, d_filter);

    cufftExecC2C(plan, (cufftComplex *)d_image,      (cufftComplex *)d_image,      CUFFT_INVERSE);
//    fftshift_1D <<<numBlocks, blockSize>>>(d_image);

    // Free filter memory, as we no longer need it.
    cudaFree(d_filter);
// ============================================================================
// Convert Fresnel propagation into hologram
    float* h_holo;
    h_holo = new float[numPix];

    float* d_holo;
    cudaMalloc((void**)&d_holo, sizeof(float)*numPix);

    createHolo <<<numBlocks, blockSize>>>(d_image, d_holo);


    cudaMemcpy(h_holo, d_holo, sizeof(float)*numPix, cudaMemcpyDeviceToHost);

//    float* myMax = std::max_element(h_holo, h_holo+(numPix-1));
    cimg_library::CImg<float> holoImg(width, height, 1, 1, 0);
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++) {
            holoImg(j, i, 0) = h_holo[i*width+j];
        }
    }


    cimg_library::CImgDisplay finalDisp(holoImg,"Hologram Image");
    while (!finalDisp.is_closed()) {
        finalDisp.wait();
    }
//    img.save_bmp("origLena.bmp");
    holoImg.save_bmp("createdHologram.png");

    cudaFree(d_image);
    cudaFree(d_holo);
}