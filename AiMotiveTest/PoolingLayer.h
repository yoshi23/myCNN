#pragma once
#include "Layer.h"

class PoolingLayer :
	public Layer
{
public:
	enum Method
	{
		Max,
		Average
	};
	PoolingLayer();
	PoolingLayer(const int & iSizeX, const int & iSizeY, const Method & iMethod, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY);
	~PoolingLayer();

	void feedForward(Layer * pNextLayer);
	void backPropagate(Layer * pPreviousLayer);
	void acceptInput(const std::vector<Eigen::MatrixXd>&);

private:
	Method mMethod;
	void downSample();

	void acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer);

	int mPoolingRegionX;
	int mPoolingRegionY;
};

