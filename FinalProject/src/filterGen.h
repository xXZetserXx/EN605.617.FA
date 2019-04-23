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

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <cufft.h>
#include <vector_types.h>

#include <CImg.h>

__global__ void calcRelativePosition(float2 *filt, float *zDist);
void calcSpatImpulseResponse(cimg_library::CImg<unsigned char> img);

#endif /* FILTERGEN_H_ */
