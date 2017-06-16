#pragma once
#include "Layer.h"
class ConvolutionalLayer :
	public Layer
{
public:
	ConvolutionalLayer();
	ConvolutionalLayer(int width, int height);
	~ConvolutionalLayer();

	void convolve();

	virtual void feedForward();
	virtual void backPropagate();

};

