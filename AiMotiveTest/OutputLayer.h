#pragma once
#include "Layer.h"
class OutputLayer :
	public Layer
{
public:
	OutputLayer();
	OutputLayer(const int & iSizeX, const int & iSizeY);
	~OutputLayer();

	virtual void feedForward();
	virtual void backPropagate();

};

