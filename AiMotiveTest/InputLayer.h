#pragma once
#include "Layer.h"
class InputLayer :
	public Layer
{
public:
	InputLayer();
	InputLayer(const int & iSizeX, const int & iSizeY);
	void feedForward();
	void backPropagate();


	~InputLayer();
};

