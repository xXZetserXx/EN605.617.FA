/*
 * assignment.cpp
 *
 *  Created on: Apr 29, 2019
 *      Author: zetser
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "info.hpp"

#define DEFAULT_PLATFORM 0

// You can't get a 16 element output using a filter with a 16 element input...so...20 elements it is.
#define NUM_BUFFER_ELEMENTS 20


// Function to check and handle OpenCL errors
inline void checkErr(cl_int err, const char * name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
	cl_int errNum;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id * platformIDs;
	cl_device_id *deviceIDs;
	cl_context context;
	cl_program program;
	std::vector<cl_kernel> kernels;
	std::vector<cl_mem> subBuffers;
	int *inputOutput;

	int platform = DEFAULT_PLATFORM;

	// First, select an OpenCL platform to run on.
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");

	platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);

	std::cout << "Number of platforms: \t" << numPlatforms << std::endl;

	errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");

	std::ifstream srcFile("simple.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(platformIDs[platform], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, numDevices, &deviceIDs[0], NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[platform], 0};

    context = clCreateContext(contextProperties, numDevices, deviceIDs, NULL, NULL, &errNum);
    checkErr(errNum, "clCreateContext");

    // Create program from source
    program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(program, numDevices, deviceIDs, "-I.", NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in OpenCL C source: " << std::endl;
		std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

    InfoDevice<cl_device_type>::display(deviceIDs[0], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");


    std::srand(std::time(nullptr));
    // Create input/output buffer and assign values.
    std::cout << "input vector" << std::endl;
    inputOutput= new int[NUM_BUFFER_ELEMENTS];
    for(int i=0; i<NUM_BUFFER_ELEMENTS; i++) {
    	inputOutput[i] = std::rand() % 100;
    	std::cout << " " << inputOutput[i];
    }
    std::cout << std::endl;

    // Create full buffer
    cl_mem fullBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*NUM_BUFFER_ELEMENTS, NULL, &errNum);
    checkErr(errNum, "clCreateBuffer");
//    subBuffers.push_back(fullBuffer);

    // create sub-buffers
    const int filtSize = 4;
    cl_mem buffer;
    for(int i=0; i<NUM_BUFFER_ELEMENTS-filtSize; i++) {
    	cl_buffer_region region = {i*sizeof(int), filtSize*sizeof(int)};
    	buffer = clCreateSubBuffer(fullBuffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
    	checkErr(errNum, "clCreateSubBuffer");
    	subBuffers.push_back(buffer);
    }



    cl_command_queue queue = clCreateCommandQueue(context, deviceIDs[0], 0, &errNum);
    checkErr(errNum, "clCreateCommandQueue");

    cl_kernel kernel = clCreateKernel(program, "average", &errNum);
    checkErr(errNum, "clCreateKernel(average)");


    errNum = clEnqueueWriteBuffer(queue, fullBuffer, CL_TRUE, 0, sizeof(int) * NUM_BUFFER_ELEMENTS, (void*)inputOutput, 0, NULL, NULL);



    std::vector<cl_event> events;
    // Still need to set fullBuffer input so I can get data out.
    //    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &fullBuffer);
    checkErr(errNum, "clSetKernelArg(average)");

    cl_event event;
	size_t gWI = filtSize;
    for(int i=0; i<NUM_BUFFER_ELEMENTS-filtSize; i++) {
    	/* I guess I need to set the subBuffer in every kernel call? simple.cpp really didn't do
    	 * anything with the subBuffers...it didn't create any unless you have multiple devices...
    	 * I don't really see the point of using subBuffers for this, it would be simpler to do it
    	 * with only the input buffer. Aren't we doing more memory copies this way?
    	 */
    	// set Input data as sub-buffer
    	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&subBuffers[i]);
    	checkErr(errNum, "clSetKernelArg(average)");

		// Queue kernel
    	errNum = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (const size_t*)&gWI,
    			                        (const size_t*)NULL, 0, 0, &event);

    	events.push_back(event);
    }
    // Wait for events to complete
    clWaitForEvents(events.size(), &events[0]);

    // Copy back data from device
    clEnqueueReadBuffer(queue, fullBuffer, CL_TRUE, 0, sizeof(int) * NUM_BUFFER_ELEMENTS,
    					(void*)inputOutput, 0, NULL, NULL);

    // We're using ints so we'll lose some precision but that's okay.
    std::cout << "Average values" << std::endl;
    for(unsigned i=0; i<NUM_BUFFER_ELEMENTS-filtSize; i++) {
    	std::cout << " " << inputOutput[i];
    }

    std::cout << "\nProgram completed successfully" << std::endl;
}
