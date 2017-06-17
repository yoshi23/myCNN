#pragma once
#include "Layer.h"
#include <vector>
#include "Dense"
class ConvolutionalLayer :
	public Layer
{
public:
	ConvolutionalLayer();
	ConvolutionalLayer(const int & iWidth, const int & iHeight, const int & iNumOfLayers, const int & iFilterWidth, const int & iFilterHeight);
	~ConvolutionalLayer();

	void convolve();

	virtual Eigen::MatrixXd feedForward();
	virtual Eigen::MatrixXd backPropagate();

private:

	std::vector<Eigen::MatrixXd> mKernels;

};

