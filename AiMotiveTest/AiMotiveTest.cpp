// AiMotiveTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "bitmap_image.hpp"
#include "IoHandler.h"
//#include "Dense"


#include "Network.h"



int main()
{




	Network wNetwork;
	wNetwork.initialize("..\\NetworkDescription.config", 52, 52); //image size 52 can be easily made dynamic of course.

	IoHandler wIoHandler;

	IoHandler::rgbPixelMap inputImage = wIoHandler.loadImage("..\\images\\train52\\6\\6_0093.bmp");

	/*for (int i = 0; i < 52; ++i) {
		for (int j = 0; j < 52; ++j) {
			std::cout << inputImage.at('r')(j, i) << "," << inputImage.at('g')(j, i) << "," << inputImage.at('b')(j, i) << std::endl;
			std::cout << "******\n";
			//std::cout << retPixelsR(j, i) << "," << retPixelsG(j, i) << "," << retPixelsB(j, i) << std::endl;
		}
	}*/
	//wIoHandler.writePixelMapToFile(wIoHandler.loadImage("..\\images\\train52\\6\\6_0093.bmp"));

	if (inputImage.size() > 0)
	{
		wIoHandler.writePixelMapToFile(inputImage);	
	}
	else
	{
		std::cout << "file was not read properly\n";
	}

	/*bitmap_image image("..\\images\\train52\\1\\1_0001.bmp");
	//bitmap_image image("..\\images\\train52\\1\\output.bmp");
	

	if (!image)
	{
		printf("Error - Failed to open: input.bmp\n");
		return 1;
	}

	unsigned int total_number_of_pixels = 0;

	const unsigned int height = image.height();
	const unsigned int width = image.width();

	for (std::size_t y = 0; y < height; ++y)
	{
		for (std::size_t x = 0; x < width; ++x)
		{
			rgb_t colour;

			image.get_pixel(x, y, colour);

			if (colour.red >= 111)
				total_number_of_pixels++;
		}
	}

	printf("Number of pixels with red >= 111: %d\n", total_number_of_pixels);*/
	

	/*Eigen::MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	//std::cout<<std::endl << m << std::endl;

	Eigen::MatrixXd n(3, 2);
	Eigen::MatrixXd na(0,0);
	n(0, 0) = 1;
	n(1, 0) = 0;
	n(0, 1) = 0;
	n(1, 1) = 1;*/
//	std::cout << std::endl << m.*n << std::endl;

    return 0;
}

