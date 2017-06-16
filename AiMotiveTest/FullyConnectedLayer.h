#pragma once
#include "Layer.h"
class FullyConnectedLayer :
	public Layer
{
public:
	FullyConnectedLayer();
	FullyConnectedLayer(const int & iSizeX, const int & iSizeY);
	~FullyConnectedLayer();

	virtual void feedForward();
	virtual void backPropagate();

};

