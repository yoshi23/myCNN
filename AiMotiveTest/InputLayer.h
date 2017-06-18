#pragma once
#include "Layer.h"
#include "Dense"


class InputLayer :
	public Layer
{
public:
	InputLayer();
	~InputLayer();
	void acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer);
	InputLayer(const int & iSizeX, const int & iSizeY);
	void feedForward(Layer * pNextLayer);
	void backPropagate(Layer * pPreviousLayer);

	void acceptInput(const std::vector<Eigen::MatrixXd>&);
	
};

