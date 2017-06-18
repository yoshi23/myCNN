#pragma once
#include "FullyConnectedLayer.h"

class OutputLayer :
	public FullyConnectedLayer
{
public:
	OutputLayer();
	OutputLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY);
	~OutputLayer();

	void feedForward();
	void provideOutput();
};

