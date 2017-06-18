// AiMotiveTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "bitmap_image.hpp"
#include "IoHandler.h"
//#include "Dense"


#include "Network.h"

#include "ConvolutionalLayer.h"

int main()
{
	Network wNetwork;
	wNetwork.initialize("..\\NetworkDescription.config", IMAGE_WIDTH, IMAGE_WIDTH); //image size IMAGE_WIDTH can be easily made dynamic of course.
	
	int dir(1);
	while (dir <=12 &&
		-1 != wNetwork.run("..\\images\\train52\\" + std::to_string(dir) + "\\" + std::to_string(dir) + "_", dir + 1))
	{
		++dir;
	}

	//IoHandler wIoHandler;

	//IoHandler::rgbPixelMap inputImage = wIoHandler.loadImage("..\\images\\trainIMAGE_WIDTH\\6\\6_0093.bmp");

	/*if (inputImage.size() > 0)
	{
		wIoHandler.writePixelMapToFile(inputImage);	
	}
	else
	{
		std::cout << "file was not read properly\n";
	}*/

	/*bitmap_image image("..\\images\\trainIMAGE_WIDTH\\1\\1_0001.bmp");
	//bitmap_image image("..\\images\\trainIMAGE_WIDTH\\1\\output.bmp");
	

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
	

	//Eigen::MatrixXd m = Eigen::MatrixXd::Random(IMAGE_WIDTH, IMAGE_WIDTH); // (10, 10, 0);// (5, 5);
	/*m(0, 0) = 1;
	m(0, 1) = 2;
	m(0, 2) = 3;
	m(0, 3) = 4;
	m(0, 4) = 5;
	m(0, 5) = 6;
	m(1, 0) = 6;
	m(1, 1) = 7;
	m(1, 2) = 8;
	m(1, 3) = 9;
	m(1, 4) = 10;
	m(2, 0) = 11;
	m(2, 1) = 12;
	m(2, 2) = 13;
	m(2, 3) = 14;
	m(2, 4) = 15;
	m(3, 0) = 16;
	m(3, 1) = 17;
	m(3, 2) = 18;
	m(3, 3) = 19;
	m(3, 4) = 20;
	m(4, 0) = 21;
	m(4, 1) = 22;
	m(4, 2) = 23;
	m(4, 3) = 24;
	m(4, 4) = 25;*/

	//std::cout << std::endl << m << std::endl;

	/*Eigen::MatrixXd n(3, 3);
	//Eigen::MatrixXd na(0,0);
	n(0, 0) = 1;
	n(0, 1) = 2;
	n(0, 2) = 3;
	n(1, 0) = 4;
	n(1, 1) = 5;
	n(1, 2) = 6;
	n(2, 0) = 7;
	n(2, 1) = 8;
	n(2, 2) = 9;*/

	//ConvolutionalLayer wConvolutionalLayer;
	//Eigen::MatrixXd res = wConvolutionalLayer.convolution(m, n,  Layer::Full);//Layer::Valid);//

//	std::cout << "\n\n" << res << std::endl;
//	std::cout << std::endl << m.*n << std::endl;

    return 0;
}

