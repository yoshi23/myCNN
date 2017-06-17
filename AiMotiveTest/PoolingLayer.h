#pragma once
#include "Layer.h"

class PoolingLayer :
	public Layer
{
public:
	PoolingLayer();
	PoolingLayer(const int & iSizeX, const int & iSizeY);
	~PoolingLayer();

	virtual void feedForward(Layer * pNextLayer);
	virtual std::map<char, Eigen::MatrixXd> backPropagate();
	void acceptInput(const std::map<char, Eigen::MatrixXd>&);
};

