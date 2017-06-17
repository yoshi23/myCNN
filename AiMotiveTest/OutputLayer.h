#pragma once
#include "Layer.h"

class OutputLayer :
	public Layer
{
public:
	OutputLayer();
	OutputLayer(const int & iSizeX, const int & iSizeY);
	~OutputLayer();

	virtual void feedForward(Layer * pNextLayer);
	virtual std::vector< Eigen::MatrixXd> backPropagate();
	void acceptInput(const std::vector<Eigen::MatrixXd>&);
	void provideOutput();
};

