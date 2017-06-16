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
	for_each(mLayers.begin(), mLayers.end(), [](Layer * iL) 
	{
		delete iL;
	}
	);
}

void Network::initialize(const std::string & networkDescriptionFile, const int & iInputSizeX, const int & iInputSizeY)
{
	mLayers.resize(0);
	InputLayer * pInputLayer = new InputLayer(iInputSizeX, iInputSizeY);
	mLayers.push_back(pInputLayer);
	std::cout << "Input layer added with size: " << pInputLayer->getSizeX() << "x" << pInputLayer->getSizeY() << " as the " << mLayers.size() << "st layer.\n";


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
				int wNewLayerWidth = wPrevLayerWidth - itPrescripedLayer->second.second + 1;
				int wNewLayerHeight = wPrevLayerHeight - itPrescripedLayer->second.first + 1;
				ConvolutionalLayer * pNewLayer = new ConvolutionalLayer(wNewLayerWidth, wNewLayerHeight);
				mLayers.push_back(pNewLayer);

				std::cout << "Convolutional layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() << "th layer.\n";

				break;
			}
			case NetworkDescriptor::Pooling:
			{
				PoolingLayer * pNewLayer = new PoolingLayer(itPrescripedLayer->second.first, itPrescripedLayer->second.second);
				mLayers.push_back(pNewLayer);
				std::cout << "Pooling layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() << "th layer.\n";
				break;
			}
			case NetworkDescriptor::FullyConnected:
			{
				FullyConnectedLayer * pNewLayer = new FullyConnectedLayer(itPrescripedLayer->second.first, itPrescripedLayer->second.second);
				mLayers.push_back(pNewLayer);
				std::cout << "FullyConnected layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() << "th layer.\n";
				break;
			}	
			case NetworkDescriptor::Output:
			{
				OutputLayer * pNewLayer = new OutputLayer(itPrescripedLayer->second.first, itPrescripedLayer->second.second);
				mLayers.push_back(pNewLayer);
				std::cout << "Output layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() << "th, last layer.\n";
				break;
			}
		}

	}
}
