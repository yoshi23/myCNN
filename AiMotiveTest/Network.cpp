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
#include <string>
#include <sstream>
#include <iomanip>

Network::Network()
{
	mRunningMode = Learning; // Working;
}


Network::~Network()
{
	for_each(mLayers.begin(), mLayers.end(), [](Layer * pLayer) 
		{
			delete pLayer;
		}
	);
}

bool Network::isLearning()
{
	if (mRunningMode == Learning) return true;
	else return false;
}

void Network::initialize(const std::string & iNetworkDescriptionFile, const int & iInputSizeX, const int & iInputSizeY)
{
	mLayers.resize(0);
	InputLayer * pInputLayer = new InputLayer(iInputSizeX, iInputSizeY);
	mLayers.push_back(pInputLayer);
	std::cout << "Input layer added with size: " << pInputLayer->getSizeX() << "x" << pInputLayer->getSizeY() << " as the 0th layer.\n";


	NetworkDescriptor wNetworkDescriptor;
	wNetworkDescriptor.readDescription(iNetworkDescriptionFile);

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


int Network::run(const std::string & iDirectory, const int & dirNum)
{
	std::cout << "dirnum: " << dirNum << std::endl;

	std::list<Layer*>::iterator it = mLayers.begin();
	IoHandler wIoHandler;

	Eigen::MatrixXd wExpectedOutput = Eigen::MatrixXd::Zero(mLayers.back()->getSizeX(), 1);
	wExpectedOutput(dirNum - 1, 0) = 1;

	std::cout << "\nExpected output: " << wExpectedOutput << "\n\n\n";

	for (int imCount = 0; imCount < 1000; ++imCount)
	{
		std::stringstream sst;
		sst << std::setfill('0') << std::setw(4) << imCount;
		
		std::string ordNum;
		sst >> ordNum;
		std::string imName = iDirectory + ordNum + ".bmp";
		IoHandler::rgbPixelMap inputImage = wIoHandler.loadImage(imName);
		//std::cout << "\nfile name to open: " << imName << std::endl << std::endl;
		if (dynamic_cast<InputLayer*>(*it) != NULL)
			(*it)->acceptInput(inputImage);
		else
		{
			std::cout << "Error: first layer of network is not an input layer. Abort.\n";
			return -1;
		}

		Layer * pCurrentLayer = *it;

		while (++it != mLayers.end())
		{
			Layer * pNextLayer = *it;
			pCurrentLayer->feedForward(pNextLayer);
			pCurrentLayer = *it;
		}
		--it;
		dynamic_cast<OutputLayer*>(*it)->feedForward(wExpectedOutput);


		//If learning is switched on then we will backgpropagate the error
		//and update the weights accordingly.
		if (mRunningMode == Network::Learning)
		{
			pCurrentLayer = *it;
			--it;
			Layer * pPreviousLayer = *it;
			dynamic_cast<OutputLayer*>(pCurrentLayer)->backPropagate(pPreviousLayer, wExpectedOutput);

			while (it != mLayers.begin())
			{
				pCurrentLayer = *it;
				--it;
				pPreviousLayer = *it;
				pCurrentLayer->backPropagate(pPreviousLayer);
			}


		}

		it = mLayers.begin();

	}
	
	return 0;
}