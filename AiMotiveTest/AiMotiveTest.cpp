// AiMotiveTest.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include "Network.h"


int main()
{
	Network wNetwork;
	wNetwork.initialize(1); //image size IMAGE_WIDTH can be easily made dynamic of course.
	
	int dir(4);
	while (
		dir <= 12 &&
		wNetwork.run("..\\images\\train52\\" + std::to_string(dir) + "\\" + std::to_string(dir) + "_", dir++) == 0
		);

    return 0;
}

