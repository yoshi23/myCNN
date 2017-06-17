#pragma once
#include "Layer.h"

class PoolingLayer :
	public Layer
{
public:
	PoolingLayer();
	PoolingLayer(const int & iSizeX, const int & iSizeY);
	~PoolingLayer();

	virtual Eigen::MatrixXd feedForward();
	virtual Eigen::MatrixXd backPropagate();
};

