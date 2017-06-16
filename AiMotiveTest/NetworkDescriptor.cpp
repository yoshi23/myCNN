#include "stdafx.h"
#include "NetworkDescriptor.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

NetworkDescriptor::NetworkDescriptor()
{
	mStructure.resize(0);
}


NetworkDescriptor::~NetworkDescriptor()
{
}

void NetworkDescriptor::readDescription(const std::string & iFileName)
{
	std::ifstream configFile(iFileName);
	std::string line;
	std::stringstream sst;
	std::string words;
	char comma;

	std::string layerType;
	int ord = 0, m = 0, n = 0;

	if (configFile.is_open())
	{
		while (getline(configFile, line))
		{
			typeAndSize newLayer;
			if (line.size()>0 && line[0] != '#')
			{
				sst << line;
				sst >> ord;
				sst >> layerType;

				switch (layerType[0])
				{
					case 'c': newLayer.first = LayerTypes::Convolutional; break;
					case 'p': newLayer.first = LayerTypes::Pooling; break;
					case 'f': newLayer.first = LayerTypes::FullyConnected; break;
					case 'o': newLayer.first = LayerTypes::Output; break;
				}

				sst >> m;
				if (newLayer.first == Convolutional || newLayer.first == Pooling)
				{
					sst >> comma;
					sst >> n;		
					newLayer.second = std::pair<int, int>(m, n);
				}
				else
				{
					newLayer.second = std::pair<int, int>(m, 1);
				}
				mStructure.push_back(newLayer);
				//std::cout << ord<<"  /nL: "<< newLayer.first<< " / "<<" /m: "<< newLayer.second.first << " /n: "<< newLayer.second.second  <<std::endl;
				sst.str(std::string());
				sst.clear();
			}
				
		}
	}
	else
	{
		std::cout << "Opening config file failed!\n";
	}
}
