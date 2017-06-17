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
		
		switch (itPrescripedLayer->first)
		{
			case NetworkDescriptor::Convolutional:
			{
				//Size of network is calculated from the kernel sizes:
				int wPrevLayerWidth = mLayers.back()->getSizeY();
				int wPrevLayerHeight = mLayers.back()->getSizeX();
				int wNumOfFilters = std::get<0>(itPrescripedLayer->second);
				int wFilterWidth = std::get<2>(itPrescripedLayer->second);
				int wFilterHeight = std::get<1>(itPrescripedLayer->second);
				int wNewLayerWidth = wPrevLayerWidth - std::get<2>(itPrescripedLayer->second) + 1;
				int wNewLayerHeight = wPrevLayerHeight - std::get<1>(itPrescripedLayer->second) + 1;
				ConvolutionalLayer * pNewLayer = new ConvolutionalLayer(wNewLayerWidth, wNewLayerHeight, wNumOfFilters, wFilterWidth, wFilterHeight);
				mLayers.push_back(pNewLayer);

				std::cout << "Convolutional layer added with size: " << wNewLayerWidth << "x" << wNewLayerHeight
					<< "\n\t\twith " << wNumOfFilters << " filters of size: " << wFilterWidth << "x" << wFilterHeight
					<< "; as the " << mLayers.size() - 1 << "th layer.\n";

				break;
			}
			case NetworkDescriptor::Pooling:
			{
				PoolingLayer * pNewLayer = new PoolingLayer(std::get<1>(itPrescripedLayer->second), std::get<2>(itPrescripedLayer->second));
				mLayers.push_back(pNewLayer);
				std::cout << "Pooling layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size()-1 << "th layer.\n";
				break;
			}
			case NetworkDescriptor::FullyConnected:
			{
				FullyConnectedLayer * pNewLayer = new FullyConnectedLayer(std::get<1>(itPrescripedLayer->second), std::get<2>(itPrescripedLayer->second));
				mLayers.push_back(pNewLayer);
				std::cout << "FullyConnected layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() - 1 << "th layer.\n";
				break;
			}	
			case NetworkDescriptor::Output:
			{
				OutputLayer * pNewLayer = new OutputLayer(std::get<1>(itPrescripedLayer->second), std::get<2>(itPrescripedLayer->second));
				mLayers.push_back(pNewLayer);
				std::cout << "Output layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() - 1 << "th, last layer.\n";
				break;
			}
		}

	}
}

/*
void run()
{
	it = mLayers.begin()
		it->receiveInput(image...);

	while (it != mLayers.back())
	{
		it->feedforward(++it);
	}
	it.provideOutput();

}*/