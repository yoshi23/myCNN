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

	virtual void feedForward(Layer * pNextLayer);
	virtual std::map<char, Eigen::MatrixXd> backPropagate();
	void acceptInput(const std::map<char, Eigen::MatrixXd>&);

private:

	std::vector<Eigen::MatrixXd> mKernels;
	double mBias;

};

