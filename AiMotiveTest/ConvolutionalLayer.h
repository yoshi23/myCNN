#pragma once
#include "Layer.h"
#include <vector>
#include "Dense"
class ConvolutionalLayer :
	public Layer
{
public:
	ConvolutionalLayer();
	ConvolutionalLayer(
		const int & iWidth,
		const int & iHeight, 
		const int & wNumOfInputFeatureMaps, 
		const int & iNumOfLayers, 
		const int & iFilterWidth, 
		const int & iFilterHeight);
	~ConvolutionalLayer();

	void convolve();

	virtual void feedForward(Layer * pNextLayer);
	virtual std::map<char, Eigen::MatrixXd> backPropagate();
	void acceptInput(const std::map<char, Eigen::MatrixXd>&);

private:

	typedef std::vector<Eigen::MatrixXd> Kernel; //these will comprise the depth of a kernel/filter
	std::vector<Kernel > mKernels;
	std::vector<Eigen::MatrixXd> mBias;

};

