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
// save diagnostic state
#pragma GCC diagnostic push
// turn off warnings
#pragma GCC diagnostic ignored "-Wvla"
#pragma GCC diagnostic ignored "-Wunreachable-code"
#include <CImg.h>
// turn the warnings back on
#pragma GCC diagnostic pop

// My headers
#include "filterGen.h"



int main(int argc, char *argv[]) {

    float zDist = .001;
    if(argc==2){
        zDist = atof(argv[1]);
    }

	std::string filename = "../pointSource.png";
	cimg_library::CImg<unsigned char> img(filename.c_str());

	std::cout << "Calling calcSpatImpulseResponse" << std::endl;
    calcSpatImpulseResponse(img, zDist);

	return 0;
}
