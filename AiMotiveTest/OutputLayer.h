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
	virtual std::map<char, Eigen::MatrixXd> backPropagate();
	void acceptInput(const std::map<char, Eigen::MatrixXd>&);

};

