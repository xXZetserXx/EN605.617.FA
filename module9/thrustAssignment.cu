/*
 * thrustAssignment.cpp
 *
 *  Created on: Apr 8, 2019
 *      Author: zetser
 */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

int main(int argc, const char **argv) {
	uint32_t N = 512*512;
	if (argc >= 2) {
		N = atoi(argv[1]);
	}


	thrust::host_vector<uint8_t> H(N);
	std::vector<uint8_t> B(N);
	for(int i=0; i<N; i++) {
		H[i] = rand();
		B[i] = H[i];
	}

	cudaEvent_t startT, stopT;
	float deltaT;


	startT = get_time();
	thrust::device_vector<int> D = H;
	thrust::sort(D.begin(), D.end());
	thrust::copy(D.begin(), D.end(), H.begin());
	stopT = get_time();


	cudaEventSynchronize(stopT);
	cudaEventElapsedTime(&deltaT, startT, stopT);

	printf("thrust::sort sorted %d size array in: %fms\n", N, deltaT);

	cudaEvent_t startT1, stopT1;
	float deltaT1;

	startT1 = get_time();
	std::sort(B.begin(), B.end());
	stopT1 = get_time();

	cudaEventElapsedTime(&deltaT1, startT1, stopT1);

	printf("std::sort sorted %d size array in: %fms\n", N, deltaT1);

	return 0;
}
