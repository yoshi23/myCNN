// AiMotiveTest.cpp : Defines the entry point for the console application.
// This project was created by Norbert Majubu (majubunorbert@gmail.com) for an interview assessment.

#include "stdafx.h"
#include "Network.h"
#include <iostream>

int main()
{
	Network wNetwork;
	int networkConfiguration; //This will determine which config file is loaded.

	std::cout << "Which config file would you like to load? (1/2/3) ";
	std::cin >> networkConfiguration;

	//This method will build up and connect the layers of the network.
	wNetwork.build(networkConfiguration); //Given more time, a builder pattern would be implemented, to make the code more flexible.
	

	//Modify this variable accordingly!
	const std::string wImageDirectory = "..\\images\\train52\\";


	//Start the already configured network. It will iterate through images in a random order.
	while (	wNetwork.run(wImageDirectory) == 0	);

    return 0;
}

