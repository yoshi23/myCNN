#include "stdafx.h"
#include "Network.h"
#include "NetworkDescriptor.h"
#include <vector>
#include "Layer.h"
#include "InputLayer.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"
#include "OutputLayer.h"

#include <iostream>

#include <algorithm>


Network::Network()
{
}


Network::~Network()
{
	for_each(mLayers.begin(), mLayers.end(), [](Layer * pLayer) 
		{
			delete pLayer;
		}
	);
}

void Network::initialize(const std::string & networkDescriptionFile, const int & iInputSizeX, const int & iInputSizeY)
{
	mLayers.resize(0);
	InputLayer * pInputLayer = new InputLayer(iInputSizeX, iInputSizeY);
	mLayers.push_back(pInputLayer);
	std::cout << "Input layer added with size: " << pInputLayer->getSizeX() << "x" << pInputLayer->getSizeY() << " as the 0th layer.\n";


	NetworkDescriptor wNetworkDescriptor;
	wNetworkDescriptor.readDescription(networkDescriptionFile);

	std::vector<NetworkDescriptor::typeAndSize>::iterator itPrescripedLayer = wNetworkDescriptor.mStructure.begin();

	for (; itPrescripedLayer != wNetworkDescriptor.mStructure.end(); ++itPrescripedLayer)
	{
		int wNumOfInputFeatureMaps = mLayers.back()->getOutPutSize();
		int wSizeOfPrevLayerX = mLayers.back()->getSizeX();
		int wSizeOfPrevLayerY = mLayers.back()->getSizeY();

		switch (itPrescripedLayer->first)
		{
			case NetworkDescriptor::Convolutional:
			{
				//Size of network is calculated from the kernel sizes:
				int wPrevLayerWidth = mLayers.back()->getSizeY();
				int wPrevLayerHeight = mLayers.back()->getSizeX();
				int wNumOfKernels = std::get<0>(itPrescripedLayer->second);
				int wKernelWidth = std::get<2>(itPrescripedLayer->second);
				int wKernelHeight = std::get<1>(itPrescripedLayer->second);
				int wNewLayerWidth = wPrevLayerWidth - std::get<2>(itPrescripedLayer->second) + 1;
				int wNewLayerHeight = wPrevLayerHeight - std::get<1>(itPrescripedLayer->second) + 1;
				ConvolutionalLayer * pNewLayer = new ConvolutionalLayer(wNewLayerWidth, wNewLayerHeight, wNumOfInputFeatureMaps, wNumOfKernels, wKernelWidth, wKernelHeight);
				mLayers.push_back(pNewLayer);

				std::cout << "Convolutional layer added with size: " << wNewLayerWidth << "x" << wNewLayerHeight
					<< "\n\t\twith " << wNumOfKernels << " Kernels of size: " << wKernelWidth << "x" << wKernelHeight
					<< "; as the " << mLayers.size() - 1 << "th layer.\n";

				break;
			}
			case NetworkDescriptor::Pooling:
			{

				PoolingLayer * pNewLayer = new PoolingLayer(std::get<1>(itPrescripedLayer->second), std::get<2>(itPrescripedLayer->second), PoolingLayer::Max, wNumOfInputFeatureMaps, wSizeOfPrevLayerX, wSizeOfPrevLayerY); //This could be easily set to be dynamicly read from config file.
				mLayers.push_back(pNewLayer);
				std::cout << "Pooling layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size()-1 << "th layer.\n";
				break;
			}
			case NetworkDescriptor::FullyConnected:
			{
				FullyConnectedLayer * pNewLayer = new FullyConnectedLayer(std::get<1>(itPrescripedLayer->second), wNumOfInputFeatureMaps, wSizeOfPrevLayerX, wSizeOfPrevLayerY);
				mLayers.push_back(pNewLayer);
				std::cout << "FullyConnected layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() - 1 << "th layer.\n";
				break;
			}	
			case NetworkDescriptor::Output:
			{
				OutputLayer * pNewLayer = new OutputLayer(std::get<1>(itPrescripedLayer->second), wNumOfInputFeatureMaps, wSizeOfPrevLayerX, wSizeOfPrevLayerY);
				mLayers.push_back(pNewLayer);
				std::cout << "Output layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() - 1 << "th, last layer.\n";
				break;
			}
			default:
			{
				break;
			}
		}

	}
}


void Network::run()
{
	std::list<Layer*>::iterator it = mLayers.begin();
	IoHandler wIoHandler;

	IoHandler::rgbPixelMap inputImage = wIoHandler.loadImage("..\\images\\train52\\6\\6_0093.bmp");

	if(dynamic_cast<InputLayer*>(*it) != NULL)
		(*it)->acceptInput(inputImage);

	Layer * wCurrentLayer = *it;
	
	while (++it != mLayers.end())
	{
		Layer * wNextLayer = *it;
		wCurrentLayer->feedForward(wNextLayer);
		wCurrentLayer = *it;
	}
	--it;
	dynamic_cast<OutputLayer*>(*it)->feedForward();

}