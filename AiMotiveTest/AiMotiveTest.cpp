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
	while (
		dir <= 12 &&
		wNetwork.run("..\\images\\train52\\" + std::to_string(dir) + "\\" + std::to_string(dir) + "_", dir++) == 0
		);

    return 0;
}

