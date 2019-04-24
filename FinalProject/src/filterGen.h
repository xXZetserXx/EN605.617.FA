/*
 * filterGen.h
 *
 *  Created on: Apr 22, 2019
 *      Author: zetser
 */

#ifndef FILTERGEN_H_
#define FILTERGEN_H_

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <cufft.h>
#include <vector_types.h>
#include <thrust/extrema.h>

// save diagnostic state
#pragma GCC diagnostic push
// turn off warnings
#pragma GCC diagnostic ignored "-Wvla"
#pragma GCC diagnostic ignored "-Wunreachable-code"
#include <CImg.h>
// turn the warnings back on
#pragma GCC diagnostic pop

typedef float2 complex;

__global__ void genFilter(float2 *filt, float *z);
void calcSpatImpulseResponse(cimg_library::CImg<unsigned char> img, float distance);

#endif /* FILTERGEN_H_ */
