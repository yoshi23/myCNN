#include "stdafx.h"
#include "Network.h"
#include "NetworkDescriptor.h"
#include <vector>
#include <set>
#include "Layer.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"
#include "OutputLayer.h"
#include "IoHandling.h"
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

#include <cstdlib>
#include <time.h>

Network::Network()
{
	mRunningMode = Learning;
	//mRunningMode = Working; 
	mConfiguration = 1;
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

void Network::build(const int & iConfiguration)
{
	std::srand((unsigned int)time(0)); //Random matrices are generated in the constructors of the layers, for the init values of the weights, etc.

	mConfiguration = iConfiguration; //Decides which network configuration file to load.
	//Loading the file which describes the structure of the network.
	NetworkDescriptor wNetworkDescriptor;
	std::string iNetworkDescriptionFile = "..\\NetworkDescription" + std::to_string(iConfiguration) + ".config";

	wNetworkDescriptor.readDescription(iNetworkDescriptionFile);

	double wEta = wNetworkDescriptor.mEta;
	double wEpsilon = wNetworkDescriptor.mEpsilon;

	int wNumOfInputFeatureMaps = 3;
	int wSizeOfPrevLayerX = IMAGE_HEIGHT;
	int wSizeOfPrevLayerY = IMAGE_WIDTH;

	mLayers.resize(0);
	std::vector<NetworkDescriptor::typeAndSize>::iterator itPrescripedLayer;
	//Iterate through the loaded description and add the layers that were read from the config file.
	for (itPrescripedLayer = wNetworkDescriptor.mStructure.begin(); itPrescripedLayer != wNetworkDescriptor.mStructure.end(); ++itPrescripedLayer)
	{
		switch (itPrescripedLayer->first)
		{
			case NetworkDescriptor::Convolutional:
			{
				//Size of new layer is calculated from thesize of previous layer and kernel sizes:
				int wNumOfKernels = std::get<0>(itPrescripedLayer->second);
				int wKernelWidth = std::get<2>(itPrescripedLayer->second);
				int wKernelHeight = std::get<1>(itPrescripedLayer->second);
				int wNewLayerWidth = wSizeOfPrevLayerY - std::get<2>(itPrescripedLayer->second) + 1;
				int wNewLayerHeight = wSizeOfPrevLayerX - std::get<1>(itPrescripedLayer->second) + 1;
				ConvolutionalLayer * pNewLayer = new ConvolutionalLayer(wNewLayerWidth, wNewLayerHeight, wNumOfInputFeatureMaps, wNumOfKernels, wKernelWidth, wKernelHeight, wEta);
				mLayers.push_back(pNewLayer);

				std::cout << "Convolutional layer added with size: " << wNewLayerHeight << "x" << wNewLayerWidth
					<< "\n\t\twith " << wNumOfKernels << " Kernels of size: " << wKernelWidth << "x" << wKernelHeight
					<< "; as the " << mLayers.size() - 1 << "th layer.\n";

				break;
			}
			case NetworkDescriptor::Pooling:
			{
				PoolingLayer * pNewLayer = new PoolingLayer(std::get<1>(itPrescripedLayer->second), std::get<2>(itPrescripedLayer->second), PoolingLayer::Max, wNumOfInputFeatureMaps, wSizeOfPrevLayerX, wSizeOfPrevLayerY);
				mLayers.push_back(pNewLayer);
				std::cout << "Pooling layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size()-1 << "th layer.\n";
				break;
			}
			case NetworkDescriptor::FullyConnected:
			{
				FullyConnectedLayer * pNewLayer = new FullyConnectedLayer(std::get<1>(itPrescripedLayer->second), wNumOfInputFeatureMaps, wSizeOfPrevLayerX, wSizeOfPrevLayerY, wEta);
				mLayers.push_back(pNewLayer);
				std::cout << "FullyConnected layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() - 1 << "th layer.\n";
				break;
			}	
			case NetworkDescriptor::Output:
			{
				OutputLayer * pNewLayer = new OutputLayer(std::get<1>(itPrescripedLayer->second), wNumOfInputFeatureMaps, wSizeOfPrevLayerX, wSizeOfPrevLayerY, wEta);
				mLayers.push_back(pNewLayer);
				std::cout << "Output layer added with size: " << pNewLayer->getSizeX() << "x" << pNewLayer->getSizeY() << " as the " << mLayers.size() - 1 << "th, last layer.\n";
				break;
			}
			default:
			{
				break;
			}
		}

		//saving parameters about this layer for the creation of the next layer;
		wNumOfInputFeatureMaps = mLayers.back()->getOutputDepth();
		wSizeOfPrevLayerX = mLayers.back()->getSizeX();
		wSizeOfPrevLayerY = mLayers.back()->getSizeY();
	}
}


int Network::run(const std::string & iDirectory)
{
	int wDirNum = 0;
	int imCount = 0;
	std::set<std::pair<int, int> > wAlreadyTrainedSet;
	std::set<std::pair<int, int> >::iterator itChecker = wAlreadyTrainedSet.begin();

	//Looping over random images in random directory.
	while(wAlreadyTrainedSet.size()<12*5000){

		//wDirNum = rand() % 12 + 1; //choose random directory between 1-12.
		wDirNum = rand() % 12 + 1; 
		//if (wAlreadyTrainedSet.size()>6 && rand() % 4 == 0) wDirNum = rand() % 12 + 1;
		//else wDirNum = 4;
		//wDirNum = 4;
		imCount = rand() % 5000;  //choose random image 0-4999
		//srand already initialized previously in this thread.


		itChecker = wAlreadyTrainedSet.find(std::make_pair(wDirNum, imCount));
		while (itChecker != wAlreadyTrainedSet.end())
		{
			imCount = rand() % 5000;  //srand already initialized previously in this thread.
			//wDirNum = rand() % 12 + 1;
			//if(rand()%4 == 0) wDirNum = rand() % 12 + 1;
			//else wDirNum = 4;
			//wDirNum = 4;
			wDirNum = rand() % 12 + 1;
			itChecker = wAlreadyTrainedSet.find(std::make_pair(wDirNum, imCount));
		}

		wAlreadyTrainedSet.insert(std::make_pair(wDirNum, imCount));
		std::cout << std::endl <<"INPUT: training image is from directory: #" << wDirNum << "/image #" << imCount << std::endl;


		std::list<Layer*>::iterator it = mLayers.begin();

		//Expected output is just a 1D vector with all-zero elements, except for the directory that we are currently in.
		Eigen::MatrixXd wExpectedOutput = Eigen::MatrixXd::Zero(mLayers.back()->getSizeX(), 1);
		wExpectedOutput(wDirNum - 1, 0) = 1;

		//Building the access name of the next image.
		std::stringstream sst;
		sst << std::setfill('0') << std::setw(4) << imCount;
		std::string wOrdNum;
		sst >> wOrdNum;
		std::string imName = iDirectory + std::to_string(wDirNum) + "\\" + std::to_string(wDirNum) + "_" + wOrdNum + ".bmp";

		std::cout << imName<<std::endl;

		IoHandling::rgbPixelMap inputImage = IoHandling::loadImage(imName);
		//Image is loaded
		//Normalize image
		for(int rgb = 0; rgb<3; ++rgb) 	inputImage[rgb].normalize();	


		//Feed image into the first layer of the network, the input layer.
		//If first layer is not an input layer, return with -1.
		if (dynamic_cast<ConvolutionalLayer*>(*it) != NULL) (*it)->acceptInput(inputImage);
		else
		{
			std::cout << "Error: first hidden layer of network is not a convolutional layer. Abort.\n";
			return -1;
		}

		//Entering the feedforwarding stage. Each layer gets the input first, process it and then passes on the output
		//to the next layer, which is provided here.
		Layer * pCurrentLayer = *it;
		while (++it != mLayers.end())
		{
			Layer * pNextLayer = *it;
			pCurrentLayer->feedForward(pNextLayer);
			pCurrentLayer = *it;
		}
		--it;
		dynamic_cast<OutputLayer*>(*it)->feedForward(wExpectedOutput);
		//Output layer is handled here seperately, because we the expected output is provided here.


		//IF THE TRAINING ERROR IS BELOW CERTAIN LEVEL, WE COULD WRITE THE PARAMETERS INTO A FILE, 
		//SO WE DO NOT HAVE TO TRAIN THE NETWORK AGAIN, IF WE DON'T WANT TO. NOT IMPLEMENTED YET.

		/*if (dynamic_cast<OutputLayer*>(*it)->getOutputError() < 0.02)
		{
				IoHandling::saveWeightsAndBiases(mLayers, mConfiguration);
		}*/


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
		//Iterator is set back to the beginning of the network and we can restart the whole cycle with a new image.
		it = mLayers.begin();
	}
	
	return 0;
}