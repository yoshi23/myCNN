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
		const int & iKernelWidth, 
		const int & iKernelHeight,
		const double & iEta
	);
	~ConvolutionalLayer();

	void convolve();

	virtual void feedForward(Layer * pNextLayer);
	virtual void backPropagate(Layer * pPreviousLayer);
	void acceptInput(const std::vector<Eigen::MatrixXd>&);

	void acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer);



private:

	void weightUpdate();
	void biasUpdate();

	typedef std::vector<Eigen::MatrixXd> Kernel; //these will comprise the depth of a kernel/Kernel
	std::vector<Kernel > mKernels;
	std::vector<Eigen::MatrixXd> mBias;

};

