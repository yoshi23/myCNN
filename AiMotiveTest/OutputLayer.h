#pragma once
#include "Layer.h"

class OutputLayer :
	public Layer
{
public:
	OutputLayer();
	OutputLayer(const int & iSizeX, const int & iSizeY);
	~OutputLayer();

	virtual Eigen::MatrixXd feedForward();
	virtual Eigen::MatrixXd backPropagate();

};

