#pragma once
#include <vector>

struct NetworkDescriptor
{
	enum LayerTypes
	{
		Input,
		Convolutional,
		Pooling,
		FullyConnected,
		Output
	};

	NetworkDescriptor();
	~NetworkDescriptor();

	void readDescription(const std::string & iFileName);

	typedef std::pair<LayerTypes, std::pair<int, int> > typeAndSize; //For each layer gives the <Type, Size>.
	std::vector<typeAndSize> mStructure; //Stores the order and typeAndSize information provided. 
};

