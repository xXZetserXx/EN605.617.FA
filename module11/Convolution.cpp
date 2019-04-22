//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

// Will set values using a random number generator function later.
cl_uint inputSignal[inputSignalWidth][inputSignalHeight];

const unsigned int outputSignalWidth  = 43;
const unsigned int outputSignalHeight = 43;

cl_float outputSignal[outputSignalWidth][outputSignalHeight];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

// Following filter values calculated with the following equation
// 1 - ( (sqrt(rowD^2 + colDist^2)-1) * 0.25)
cl_float mask[maskWidth][maskHeight] =
{
	{0.19, 0.35, 0.46, 0.50, 0.46, 0.35, 0.19},
	{0.35, 0.54, 0.69, 0.75, 0.69, 0.54, 0.35},
	{0.46, 0.69, 0.89, 1.00, 0.89, 0.69, 0.46},
	{0.50, 0.75, 1.00, 1.00, 1.00, 0.75, 0.50},
	{0.46, 0.69, 0.89, 1.00, 0.89, 0.69, 0.46},
	{0.35, 0.54, 0.69, 0.75, 0.69, 0.54, 0.35},
	{0.19, 0.35, 0.46, 0.50, 0.46, 0.35, 0.19}
};

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

/**
 * Function to generate random data to put into the inputSignal global array
 */
void generateRandInput() {
	srand(time(NULL));
	for(int i=0; i<inputSignalWidth; i++) {
		for(int j=0; j<inputSignalHeight; j++) {
			inputSignal[i][j] = rand() % 10 + 1;		// Generate random number between 1 & 10
		}
	}
}

///
//	main() for Convolution example
//
int run()
{
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, numDevices, &deviceIDs[0], NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
// 	if (deviceIDs == NULL) {
// 		std::cout << "No CPU device found" << std::endl;
// 		exit(-1);
// 	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(contextProperties, numDevices, deviceIDs, &contextCallback, NULL, &errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(program, numDevices, deviceIDs, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(program, "convolve", &errNum);
	checkErr(errNum, "clCreateKernel");

	// TODO: Set values in inputSignal. I guess random values works?
	generateRandInput();

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth, static_cast<void *>(inputSignal), &errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * maskHeight * maskWidth, static_cast<void *>(mask), &errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer( context, CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth, NULL, &errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(context,deviceIDs[0], 0, &errNum);
	checkErr(errNum, "clCreateCommandQueue");

	// Get Timestamp
	auto tStartCopy = std::chrono::high_resolution_clock::now();

    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), 	&inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), 	&maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), 	&outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), 	&inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), 	&maskWidth);
	checkErr(errNum, "clSetKernelArg");

	auto tStartKernel = std::chrono::high_resolution_clock::now();

	const size_t globalWorkSize[1] = { outputSignalWidth * outputSignalHeight };
    const size_t localWorkSize[1]  = { outputSignalWidth };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clFinish(queue);
	checkErr(errNum, "clEnqueueNDRangeKernel");


	auto tStartReadout = std::chrono::high_resolution_clock::now();

	errNum = clEnqueueReadBuffer(queue, outputSignalBuffer, CL_TRUE,
        0, sizeof(cl_uint) * outputSignalHeight * outputSignalHeight,
		outputSignal, 0, NULL, NULL);
	clFinish(queue);
	checkErr(errNum, "clEnqueueReadBuffer");

	auto tStopReadout = std::chrono::high_resolution_clock::now();

//    // Output the result buffer
//    for (int y = 0; y < outputSignalHeight; y++)
//	{
//		for (int x = 0; x < outputSignalWidth; x++)
//		{
//			std::cout << outputSignal[x][y] << " ";
//		}
//		std::cout << std::endl;
//	}

    std::cout << std::endl << "Executed program successfully." << std::endl;

    auto copy2Dev = std::chrono::duration_cast<std::chrono::duration<double>>(tStartKernel  - tStartCopy);
	auto kernelT  = std::chrono::duration_cast<std::chrono::duration<double>>(tStartReadout - tStartKernel);
	auto readOut  = std::chrono::duration_cast<std::chrono::duration<double>>(tStopReadout  - tStartReadout);
	auto totalT   = std::chrono::duration_cast<std::chrono::duration<double>>(tStopReadout  - tStartKernel);

//	printf("Array Size: %d\t Size in Bytes: %d\n", 	ARRAY_SIZE, ARRAY_SIZE*sizeof(float)/8);
	printf("Time to copy data to device: %fms\n", 					static_cast<float>(copy2Dev.count()*1000));
	printf("Time to run the kernels: %fms\n", 						static_cast<float>(kernelT.count()*1000));
	printf("Time to read data back to host: %fms\n", 				static_cast<float>(readOut.count()*1000));
	printf("Total time of kernel execution and copies: %fms\n\n", 	static_cast<float>(totalT.count()*1000));

	return 0;
}

int main(int argc, char** argv) {
	run();
	printf("========================================================\n");
	run();

	return 0;
}
