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
	retMap.insert(std::make_pair('r', retPixelsR));
	retMap.insert(std::make_pair('g', retPixelsG));
	retMap.insert(std::make_pair('b', retPixelsB));

	return retMap;
}


void IoHandler::writePixelMapToFile(const IoHandler::rgbPixelMap & iImage){

	FILE *fr = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedR.txt", "w");
	FILE *fg = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedG.txt", "w");
	FILE *fb = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedB.txt", "w");
	IoHandler::rgbPixelMap::const_iterator it = iImage.begin();
	
	for (int i = 0; i < IMAGE_WIDTH; ++i) {
		for (int j = 0; j < 51; ++j) {
			fprintf(fr, "%d, ", static_cast<int>(iImage.at('r')(i, j))); 
			fprintf(fg, "%d, ", static_cast<int>(iImage.at('g')(i, j)));
			fprintf(fb, "%d, ", static_cast<int>(iImage.at('b')(i, j)));
			--it;
			--it;
		}
		fprintf(fr, "%d ", iImage.at('r')(i, 51));
		fprintf(fg, "%d ", iImage.at('g')(i, 51));
		fprintf(fb, "%d ", iImage.at('b')(i, 51));
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