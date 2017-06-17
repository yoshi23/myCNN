#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include <algorithm>

ConvolutionalLayer::ConvolutionalLayer()
{
}

ConvolutionalLayer::ConvolutionalLayer(const int & iWidth, const int & iHeight, const int & iNumOfLayers, const int & iFilterWidth, const int & iFilterHeight)
{
	mSizeX = iHeight;
	mSizeY = iWidth;
}


ConvolutionalLayer::~ConvolutionalLayer()
{
}

void ConvolutionalLayer::convolve()
{
	for_each(mKernels.begin(), mKernels.end(), [](Eigen::MatrixXd & Kernel)
		{
			
		}
	);
		



}

void ConvolutionalLayer::feedForward(Layer * pNextLayer)
{
	convolve();
	pNextLayer->acceptInput(mOutput);
}

std::map<char, Eigen::MatrixXd> ConvolutionalLayer::backPropagate()
{//MOCK
	std::map<char, Eigen::MatrixXd> fake;
	return fake;
}


void ConvolutionalLayer::acceptInput(const std::map<char, Eigen::MatrixXd>& iInputMap)
{
	mInput = iInputMap;
}