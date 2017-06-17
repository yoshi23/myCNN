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

	void feedForward(Layer * pNextLayer);
	virtual std::map<char, Eigen::MatrixXd> backPropagate();
	void acceptInput(const std::map<char, Eigen::MatrixXd>&);

};

