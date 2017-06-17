#pragma once
#include "Layer.h"

class PoolingLayer :
	public Layer
{
public:
	PoolingLayer();
	PoolingLayer(const int & iSizeX, const int & iSizeY);
	~PoolingLayer();

	void feedForward(Layer * pNextLayer);
	std::vector<Eigen::MatrixXd> backPropagate();
	void acceptInput(const std::vector<Eigen::MatrixXd>&);
};

