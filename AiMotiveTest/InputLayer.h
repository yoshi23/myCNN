#pragma once
#include "Layer.h"
#include "Dense"
#include "IoHandler.h"
#include <map>

class InputLayer :
	public Layer
{
public:
	InputLayer();
	~InputLayer();
	InputLayer(const int & iSizeX, const int & iSizeY);
	void feedForward(Layer * pNextLayer);
	std::map<char, Eigen::MatrixXd> backPropagate();


	void acceptInput(const std::map<char, Eigen::MatrixXd>&);
	
};

