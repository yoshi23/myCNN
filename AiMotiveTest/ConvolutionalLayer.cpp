#include "stdafx.h"
#include "ConvolutionalLayer.h"


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




}

Eigen::MatrixXd ConvolutionalLayer::feedForward()
{//mock
	return Eigen::MatrixXd(1,1);
}

Eigen::MatrixXd ConvolutionalLayer::backPropagate()
{//MOCK
	return Eigen::MatrixXd(1, 1);
}
