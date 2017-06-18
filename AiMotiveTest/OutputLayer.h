#pragma once
#include "FullyConnectedLayer.h"

class OutputLayer :
	public FullyConnectedLayer
{
public:
	OutputLayer();
	OutputLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY);
	~OutputLayer();

	void feedForward();
	//virtual void feedForward(Layer * pNextLayer);
	//virtual std::vector< Eigen::MatrixXd> backPropagate();
	//void acceptInput(const std::vector<Eigen::MatrixXd>&);
	void provideOutput();
};

