#include "stdafx.h"
#include "NetworkDescriptor.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>

NetworkDescriptor::NetworkDescriptor()
{
	mStructure.resize(0);
}


NetworkDescriptor::~NetworkDescriptor()
{
}

//This method reads data from config files.
void NetworkDescriptor::readDescription(const std::string & iFileName)
{
	std::ifstream configFile(iFileName);
	std::string line;
	std::stringstream sst;
	std::string words;
	std::string symbols;
	char separationSymbols;

	std::string layerType;
	int ord = 0, m = 0, n = 0;
	int numOfKernels = 0;

	if (configFile.is_open())
	{
		while (getline(configFile, line))
		{

			if (line.size() > 0 && line[0] == '*') //it means it is a algorithmic parameter for the network
			{
				sst << line;
				sst >> separationSymbols;
				sst >> symbols;
				if (symbols == "ETA")
				{
					sst >> mEta;
				}
				sst.str(std::string());
				sst.clear();
			}

			typeAndSize newLayer;
			if (line.size()>0 && line[0] != '#' && line[0] != '*')
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
		configFile.close();
	}
	else
	{
		std::cout << "Opening config file failed!\n";
	}

}
