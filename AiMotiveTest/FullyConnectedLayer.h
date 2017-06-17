#pragma once
#include "Layer.h"
//#include "Dense"
#include <vector>
class FullyConnectedLayer :
	public Layer
{
public:
	FullyConnectedLayer();
	FullyConnectedLayer(const int & iSizeX, const int & iSizeY);
	~FullyConnectedLayer();

	void feedForward(Layer * pNextLayer);
	virtual std::vector< Eigen::MatrixXd> backPropagate();
	void acceptInput(const std::vector<Eigen::MatrixXd>&);

};

