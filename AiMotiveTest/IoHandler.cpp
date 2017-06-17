#include "stdafx.h"
#include "IoHandler.h"

#include "SFML\Graphics\Image.hpp"
#include <iostream>
#include <map>

#include <cstdio>

#include "Dense"
#include "Layer.h"

IoHandler::IoHandler()
{
}


IoHandler::~IoHandler()
{
}

IoHandler::rgbPixelMap IoHandler::loadImage(const std::string & iFileName)
{
	sf::Image wImageFile;
	if (!wImageFile.loadFromFile(iFileName))
		return rgbPixelMap();

	sf::Color pixel;

	Eigen::MatrixXd retPixelsR(IMAGE_WIDTH, IMAGE_HEIGHT);
	Eigen::MatrixXd retPixelsG(IMAGE_WIDTH, IMAGE_HEIGHT);
	Eigen::MatrixXd retPixelsB(IMAGE_WIDTH, IMAGE_HEIGHT);

	for (int i = 0; i < wImageFile.getSize().x; ++i) {
		for (int j = 0; j < wImageFile.getSize().y; ++j) {
			pixel = wImageFile.getPixel(j, i);

			retPixelsR(i, j) = static_cast<int>(pixel.r);
			retPixelsG(i, j) = static_cast<int>(pixel.g);
			retPixelsB(i, j) = static_cast<int>(pixel.b);
		}
	}

	rgbPixelMap retMap;
	retMap.push_back(retPixelsR);
	retMap.push_back(retPixelsG);
	retMap.push_back(retPixelsB);

	return retMap;
}


void IoHandler::writePixelMapToFile(const IoHandler::rgbPixelMap & iImage){

	FILE *fr = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedR.txt", "w");
	FILE *fg = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedG.txt", "w");
	FILE *fb = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedB.txt", "w");
	IoHandler::rgbPixelMap::const_iterator it = iImage.begin();
	
	for (int i = 0; i < IMAGE_WIDTH; ++i) {
		for (int j = 0; j < IMAGE_HEIGHT - 1; ++j) {
			fprintf(fr, "%d, ", static_cast<int>(iImage[0](i, j))); 
			fprintf(fg, "%d, ", static_cast<int>(iImage[1](i, j)));
			fprintf(fb, "%d, ", static_cast<int>(iImage[2](i, j)));
			--it;
			--it;
		}
		fprintf(fr, "%d ", iImage[0](i, 51));
		fprintf(fg, "%d ", iImage[1](i, 51));
		fprintf(fb, "%d ", iImage[2](i, 51));
		--it;
		--it;
		fprintf(fr, "\n");
		fprintf(fg, "\n");
		fprintf(fb, "\n");
	}

	fclose(fr);
	fclose(fg);
	fclose(fb);
	std::cout << "Writing to file finished\n";
}