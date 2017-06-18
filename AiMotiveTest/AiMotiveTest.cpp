// AiMotiveTest.cpp : Defines the entry point for the console application.
// This project was created by Norbert Majubu (majubunorbert@gmail.com) for an interview assessment.

#include "stdafx.h"
#include "Network.h"

int main()
{
	Network wNetwork;

	int networkConfiguration = 1; //This will determine which config file is loaded.

	//This method will build up and connect the layers of the network.
	wNetwork.initialize(networkConfiguration); //Given more time, a builder pattern would be implemented, to make the code more flexible.
	
	//Here, we iterate through the 12 libraries that are present in the training database.
	//For each directory the run method of Network is called and it iterates through the images in the directory.
	int dir(1);
	while (
		dir <= 12 &&
		wNetwork.run("..\\images\\train52\\" + std::to_string(dir) + "\\" + std::to_string(dir) + "_") == 0
		);

    return 0;
}

