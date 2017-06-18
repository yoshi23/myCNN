#include "stdafx.h"
#include "OutputLayer.h"
#include <iostream>
#include <map>
#include <vector>

OutputLayer::OutputLayer()
{
}

OutputLayer::OutputLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY)
	: FullyConnectedLayer(iSizeX, iNumOfInputFeatureMaps, iSizeOfPrevLayerX,  iSizeOfPrevLayerY)
{
}

OutputLayer::~OutputLayer()
{
}

void OutputLayer::provideOutput()
{
	std::cout << "Network forward propagation has reached the end of the network! Hurray!\n";
	std::cout << mOutput[0];
}

void OutputLayer::feedForward()
{
	calculateActivation();
	provideOutput();
}