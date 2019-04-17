/*
 * main.cpp
 *
 *  Created on: Apr 16, 2019
 *      Author: Mikhail Wolff
 */


#include <string>

#include <CImg.h>


int main(int argc, char *argv[]) {

	std::string filename = "Lena.png";
	cimg_library::CImg<unsigned char> img(filename.c_str());

	int width = img.width();
	int height = img.height();

	// TODO: Generate filter kernel image.

	return 0;
}
