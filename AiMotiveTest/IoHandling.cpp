#include "stdafx.h"
#include "IoHandling.h"
#include "SFML\Graphics\Image.hpp"
#include <iostream>
#include <list>
#include <cstdio>
#include "Dense"
#include "Layer.h"
#include "ConvolutionalLayer.h"
#include "FullyConnectedLayer.h"
#include <fstream>

namespace IoHandling
{
	rgbPixelMap loadImage(const std::string & iFileName)
	{
		sf::Image wImageFile;
		if (!wImageFile.loadFromFile(iFileName))
			return rgbPixelMap();

		sf::Color pixel;

		Eigen::MatrixXd retPixelsR(IMAGE_WIDTH, IMAGE_HEIGHT);
		Eigen::MatrixXd retPixelsG(IMAGE_WIDTH, IMAGE_HEIGHT);
		Eigen::MatrixXd retPixelsB(IMAGE_WIDTH, IMAGE_HEIGHT);

		for (unsigned int i = 0; i < wImageFile.getSize().x; ++i) {
			for (unsigned int j = 0; j < wImageFile.getSize().y; ++j) {
				pixel = wImageFile.getPixel(j, i);

				retPixelsR(i, j) = static_cast<int>(pixel.r);
				retPixelsG(i, j) = static_cast<int>(pixel.g);
				retPixelsB(i, j) = static_cast<int>(pixel.b);
			}
		}

		rgbPixelMap retMap;
		retMap.push_back(retPixelsR);
		retMap.push_back(retPixelsG);
		retMap.push_back(retPixelsB);

		return retMap;
	}

	//TODO: I WANT TO USE THE SERIALIZATION LIBRARY OF BOOST INSTEAD OF THIS. CURRENTLY NOT DONE WITH THAT.
	void saveWeightsAndBiases(const std::list<Layer*> & iLayers, const int & ID)
	{
	}

	//TODO: NOT IMPLEMENTED YET.
	void loadWeightsAndBiases(Layer * iLayer)
	{
	}

	//writes the name of the road sign explicitly and information about the error rate.
	void nameTable(const std::vector<Eigen::MatrixXd> & iOuptut, const double & iError, const Eigen::MatrixXd & iExpectedOutput)
	{
		std::string wFileName = "..\\Error_rate.csv";
		std::ofstream errorFile(wFileName, std::ofstream::out | std::ofstream::app);

		std::vector<std::string> SignTypes =
		{
			"Dangerous Cliff",
			"Road Works",
			"Stop Sign",
			"End Of Main Route",
			"No Entry",
			"Danger from Above",
			"Speedlimit 130",
			"No Stopping",
			"Roundabout",
			"Pedestrian route",
			"Highway",
			"Crosswalk"
		};

		int maxInd(0);
		int explicitHit(0);
		for (int i = 1; i < iOuptut[0].rows(); ++i)
		{
			if (iExpectedOutput(i, 0) == 1)
			{
				explicitHit = i;
			}
			if (iOuptut[0](i, 0) > iOuptut[0](maxInd, 0))
			{
				maxInd = i;
			}
		}
		
		std::cout << iOuptut[0] << std::endl;
		std::cout << "Error rate: " << iError << std::endl;
		std::cout << "DECISION: This is a dir #" + std::to_string(maxInd + 1) + " picture, a(n) "+ SignTypes[maxInd] + " sign.\n";
		if (maxInd == explicitHit)
		{
			std::cout << "PERFECT DECISION\n";
			errorFile << iError << ", " << 1 << std::endl;
		}
		else
		{
			std::cout << "BAD DECISION\n";
			errorFile << iError << ", " << 0 << std::endl;
		}

	}
}