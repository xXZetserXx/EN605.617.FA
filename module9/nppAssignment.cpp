/*
 * nppAssignment.cu
 *
 *  Created on: Apr 8, 2019
 *      Author: wolffmb1
 */
#include "CImg.h"

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

#define NUMCHAN 3

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

// Taken from boxFilterNPP.cpp
bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

std::string getFilename(int argc, char *argv[]) {

	std::string sFilename;
	char *filePath;
	if (checkCmdLineFlag(argc, (const char **)argv, "input"))
	{
		getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
	}
	else
	{
		filePath = sdkFindFilePath("Lena.bmp", argv[0]);
	}

	if (filePath)
	{
		sFilename = filePath;
	}
	else
	{
		sFilename = "../common/data/Lena.bmp";
	}

	int file_errors = 0;
	std::ifstream infile(sFilename.data(), std::ifstream::in);
	if (infile.good())
	{
		std::cout << "opened: <" << sFilename.data() << "> successfully!" << std::endl;
		file_errors = 0;
		infile.close();
	}
	else
	{
		std::cout << "unable to open: <" << sFilename.data() << ">" << std::endl;
		file_errors++;
		infile.close();
	}

	if (file_errors > 0)
	{
		exit(EXIT_FAILURE);
	}
	return sFilename;
}

// This function will just deal with calling NPP, the image should already be loaded into an array of type Npp8u
// image size is fixed for the time being.
void callNpp(Npp8u *img, int width, int height) {
	const int num_pixels = width*height;
	const int imgWidth = width*sizeof(Npp8u)*NUMCHAN;
	const int imgSize = num_pixels*sizeof(Npp8u)*NUMCHAN;
	const int filtSize = 9*sizeof(Npp32f);

	// Sharpening filter
	const Npp32f filter[9] = {-1.0, -1.0, -1.0,
							  -1.0,  9.0, -1.0,
							  -1.0, -1.0, -1.0};

	Npp8u *d_inImg, *d_outImg;
	Npp32f *d_filt;
//	// Allocate space for input and output variables.
	cudaMalloc(&d_inImg,	imgSize);
	cudaMalloc(&d_outImg,	imgSize);
	cudaMalloc(&d_filt,		filtSize);
//	// Copy input data to device.
	cudaMemcpy(d_inImg,   img,	imgSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filt, filter, filtSize, cudaMemcpyHostToDevice);

	const NppiSize oKernelSize = {3, 3};
	const NppiPoint oAnchor = {1, 1};
	const NppiSize oSrcSize = {width, height};
	const NppiPoint oSrcOffset = {0, 0};
	const NppiSize oSizeROI = {width, height};

	// Device output image
//	npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

	std::cout << "Executing filter..." << std::endl;
	nppiFilterBorder32f_8u_C3R(d_inImg, imgWidth, oSrcSize, oSrcOffset,
											 d_outImg, imgWidth, oSizeROI,
											 d_filt, oKernelSize, oAnchor, NPP_BORDER_REPLICATE);

	cudaMemcpy(img, d_outImg, imgSize, cudaMemcpyDeviceToHost);
	cudaFree(d_inImg);
	cudaFree(d_outImg);
	cudaFree(d_filt);
}


int main(int argc, char *argv[]) {


	std::string sFilename = getFilename(argc, argv);
	cimg_library::CImg<unsigned char> img(sFilename.c_str());
	int width = img.width();
	int height = img.height();

	Npp8u *nppImg = (Npp8u*)malloc(NUMCHAN*width*height*sizeof(Npp8u));

	// Copy image data over to Npp8u object
	for(int i=0; i<height; i++) {
		for(int j=0; j<width; j++) {
			for(int k=0; k<NUMCHAN; k++) {
				nppImg[NUMCHAN*(i*width+j)+k] = img(j, i, k);
			}
		}
	}

	std::cout << "Running callNpp..." << std::endl;
	callNpp(nppImg, width, height);

	// Copy image data over to Npp8u object
	for(int i=0; i<height; i++) {
		for(int j=0; j<width; j++) {
			for(int k=0; k<NUMCHAN; k++) {
				img(j, i, k) = nppImg[NUMCHAN*(i*width+j)+k];
			}
		}
	}

	img.save_bmp("nppOut.bmp");
	std::cout << "Image saved to nppOut.bmp" << std::endl;


	return 0;
}
