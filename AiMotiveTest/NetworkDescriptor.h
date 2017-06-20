#pragma once
#include <vector>
#include <tuple>

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

	typedef std::pair<LayerTypes, std::tuple<int, int, int> > typeAndSize; //For each layer gives the <Type, <num of Kernels, Size> >.
	std::vector<typeAndSize> mStructure; //Stores the order and typeAndSize information provided. 

	double mEta; //updating speed of learning of network
	double mEpsilon; //for calculating activation gradients

};

