#pragma once
#include "Layer.h"
//#include "Dense"
class FullyConnectedLayer :
	public Layer
{
public:
	FullyConnectedLayer();
	FullyConnectedLayer(const int & iSizeX, const int & iSizeY);
	~FullyConnectedLayer();

	virtual Eigen::MatrixXd feedForward();
	virtual Eigen::MatrixXd backPropagate();

};

