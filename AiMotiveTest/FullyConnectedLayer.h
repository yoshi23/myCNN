#pragma once
#include "Layer.h"
//#include "Dense"
#include <vector>
class FullyConnectedLayer :
	public Layer
{
public:
	FullyConnectedLayer();
	FullyConnectedLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY);
	~FullyConnectedLayer();

	void feedForward(Layer * pNextLayer);
	virtual void backPropagate(Layer * pPreviousLayer);
	void acceptInput(const std::vector<Eigen::MatrixXd>&);
	

protected:
	void calculateActivation();
	void calculateActivationGradient();
	void weightUpdate();
	void biasUpdate();

	//TODO: COULD HAVE A COMMON PARENT CLASS WITH CONVOLUTIONAL LAYER!
	typedef std::vector<Eigen::MatrixXd> Weights; //these will comprise the depth of a kernel/Kernel
	std::vector<Weights > mWeights;
	std::vector<double> mBiases;

};

