#include "stdafx.h"
#include "IoHandler.h"

#include "SFML\Graphics\Image.hpp"
#include <iostream>
#include <cstdio>

IoHandler::IoHandler()
{
}


IoHandler::~IoHandler()
{
}

bool IoHandler::loadImage(const std::string iFileName)
{
	sf::Image background;
	if (!background.loadFromFile("..\\images\\train52\\6\\6_0093.bmp"))
		return false;

	sf::Color pixel;

	FILE *fr = fopen("..\\images\\train52\\matrixOutputRedR.txt", "w");
	FILE *fg = fopen("..\\images\\train52\\matrixOutputRedG.txt", "w");
	FILE *fb = fopen("..\\images\\train52\\matrixOutputRedB.txt", "w");


	for (int i = 0; i < background.getSize().x; ++i) {
		for (int j = 0; j < background.getSize().y-1; ++j) {
			pixel = background.getPixel(j, i);
			//std::cout << static_cast<int>(pixel.r) << " ";
			fprintf(fr, "%d, ", static_cast<int>(pixel.r));
			fprintf(fg, "%d, ", static_cast<int>(pixel.g));
			fprintf(fb, "%d, ", static_cast<int>(pixel.b));
		}
		pixel = background.getPixel(51, i);
		fprintf(fr, "%d ", static_cast<int>(pixel.r));
		fprintf(fg, "%d ", static_cast<int>(pixel.g));
		fprintf(fb, "%d ", static_cast<int>(pixel.b));
		fprintf(fr, "\n"); // std::cout << std::endl;
		fprintf(fg, "\n");
		fprintf(fb, "\n");
	}

	fclose(fr);
	fclose(fg);
	fclose(fb);
	//std::cout << "Writing to file is ready";


	return true;
}
