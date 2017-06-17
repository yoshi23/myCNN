#pragma once
#include "Layer.h"
#include "Dense"
class InputLayer :
	public Layer
{
public:
	InputLayer();
	InputLayer(const int & iSizeX, const int & iSizeY);
	Eigen::MatrixXd feedForward();
	Eigen::MatrixXd backPropagate();


	//void getInput()

	~InputLayer();
};

