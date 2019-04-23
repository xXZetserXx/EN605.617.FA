/*
 * main.cpp
 *
 *  Created on: Apr 16, 2019
 *      Author: Mikhail Wolff
 */

// Dev libraries
#include <iostream>
#include <string>
#include <cmath>

// External Libraries
#include <CImg.h>

// My headers
#include "filterGen.h"

typedef float2 complex;

int main(int argc, char *argv[]) {

	std::string filename = "../../common/data/Lena.bmp";
	cimg_library::CImg<unsigned char> img(filename.c_str());

	std::cout << "Calling calcSpatImpulseResponse" << std::endl;
    calcSpatImpulseResponse(img);

	return 0;
}
