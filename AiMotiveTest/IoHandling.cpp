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

		for (int i = 0; i < wImageFile.getSize().x; ++i) {
			for (int j = 0; j < wImageFile.getSize().y; ++j) {
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


	void writePixelMapToFile(const rgbPixelMap & iImage) {

		FILE *fr = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedR.txt", "w");
		FILE *fg = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedG.txt", "w");
		FILE *fb = fopen("..\\images\\trainIMAGE_WIDTH\\matrixOutputRedB.txt", "w");
		rgbPixelMap::const_iterator it = iImage.begin();

		for (int i = 0; i < IMAGE_WIDTH; ++i) {
			for (int j = 0; j < IMAGE_HEIGHT - 1; ++j) {
				fprintf(fr, "%d, ", static_cast<int>(iImage[0](i, j)));
				fprintf(fg, "%d, ", static_cast<int>(iImage[1](i, j)));
				fprintf(fb, "%d, ", static_cast<int>(iImage[2](i, j)));
				--it;
				--it;
			}
			fprintf(fr, "%d ", iImage[0](i, 51));
			fprintf(fg, "%d ", iImage[1](i, 51));
			fprintf(fb, "%d ", iImage[2](i, 51));
			--it;
			--it;
			fprintf(fr, "\n");
			fprintf(fg, "\n");
			fprintf(fb, "\n");
		}

		fclose(fr);
		fclose(fg);
		fclose(fb);
		std::cout << "Writing to file finished\n";
	}

	void saveWeightsAndBiases(const std::list<Layer*> & iLayers, const int & ID)
	{
		std::string wFileName = "SavedParameters" + std::to_string(ID) + ".csv";
		std::ofstream saveFile(wFileName);

		std::list<Layer*>::const_iterator itLayer = iLayers.begin();

		while (itLayer != iLayers.end())
		{
			if (dynamic_cast<ConvolutionalLayer*>(*itLayer) != 0)
			{/*
				for (int i = 0; i < (*itLayer)->getNumOfKernels(); ++i)
				{
					saveFile << "Kernel:\n";
					saveFile << (*itLayer)->getKernel(i);
					saveFile << "Bias:\n";
					saveFile << (*itLayer)->getBiases(i);
				}*/
				
				
			}
			else if (dynamic_cast<FullyConnectedLayer*>(*itLayer) != 0)
			{
				FullyConnectedLayer * itFull = dynamic_cast<FullyConnectedLayer*>(*itLayer);
				for (int neuron = 0; neuron < itFull->getSizeX(); ++neuron)
				{
					for (int depth = 0; depth < itFull->getDepth(); ++depth)
					{
						saveFile << "Kernel:\n";
						saveFile << itFull->getWeights(neuron, depth);
						saveFile << "Bias:\n";
						saveFile << itFull->getBiases(neuron);
					}

				
				}

			}
			
			++itLayer;

		}
		/*

		std::string line;
		std::stringstream sst;
		std::string words;
		char separationSymbols;

		std::string layerType;
		int ord = 0, m = 0, n = 0;
		int numOfKernels = 0;

		if (saveFile.is_open())
		{
			while (getline(saveFile, line))
			{
				typeAndSize newLayer;
				if (line.size()>0 && line[0] != '#')
				{
					sst << line;
					sst >> ord;
					sst >> layerType;

					switch (layerType[0])
					{
					case 'c':
					{
						newLayer.first = LayerTypes::Convolutional;
						sst >> numOfKernels;
						sst >> separationSymbols;
						break;
					}
					case 'p': newLayer.first = LayerTypes::Pooling; break;
					case 'f': newLayer.first = LayerTypes::FullyConnected; break;
					case 'o': newLayer.first = LayerTypes::Output; break;
					default: break;
					}


					sst >> m;
					if (newLayer.first == Convolutional || newLayer.first == Pooling)
					{
						sst >> separationSymbols;
						sst >> n;
						if (newLayer.first == Convolutional)
						{
							newLayer.second = std::tuple<int, int, int>(numOfKernels, m, n);
						}
						else
						{
							newLayer.second = std::tuple<int, int, int>(1, m, n);
						}

					}
					else
					{
						newLayer.second = std::tuple<int, int, int>(1, m, 1);
					}
					mStructure.push_back(newLayer);
					sst.str(std::string());
					sst.clear();
				}

			}
		}
		else
		{
			std::cout << "Opening config file failed!\n";
		}
		

		*/

	}

	void loadWeightsAndBiases(Layer * iLayer)
	{
	}


	void nameTable(const std::vector<Eigen::MatrixXd> & iOuptut, const double & iError)
	{
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

		double maxInd(0);
		for (int i = 1; i < iOuptut[0].rows(); ++i)
		{
			if (iOuptut[0](i, 0) > iOuptut[0](maxInd, 0))
			{
				maxInd = i;
			}
		}
		std::cout << iOuptut[0] << std::endl;
		std::cout << "Error rate: " << iError << std::endl;
		std::cout << "This is a " + SignTypes[maxInd] + " sign.\n";

	}

}