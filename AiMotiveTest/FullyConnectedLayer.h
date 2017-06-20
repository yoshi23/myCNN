#pragma once
#include "Layer.h"
//#include "Dense"
#include <vector>
class FullyConnectedLayer :
	public Layer
{
public:
	FullyConnectedLayer();
	FullyConnectedLayer(
		const int & iSizeX, 
		const int & iNumOfInputFeatureMaps,
		const int & iSizeOfPrevLayerX,
		const int & iSizeOfPrevLayerY,
		const double & iEta,
		const double & iEpsilon
	);
	~FullyConnectedLayer();

	void feedForward(Layer * pNextLayer);
	virtual void backPropagate(Layer * pPreviousLayer);
	void acceptInput(const std::vector<Eigen::MatrixXd>&);
	void acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer);
	
protected:

	void calculateActivation();
	void weightUpdate();
	void biasUpdate();
	void calcDeltaOfLayer();

	typedef std::vector<Eigen::MatrixXd> Weights; //these will comprise the depth of a kernel/Kernel
	std::vector<Weights > mWeights;
	std::vector<double> mBiases;

};

