#pragma once
#include "Layer.h"
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
		const double & iEta
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

	typedef std::vector<Eigen::MatrixXd> Weights; //these will comprise the depth of a kernel.
	std::vector<Weights > mWeights;
	std::vector<double> mBiases;


	//static int filecounter;

};

