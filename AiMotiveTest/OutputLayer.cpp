#include "stdafx.h"
#include "OutputLayer.h"
//#include "Dense"

OutputLayer::OutputLayer()
{
}

OutputLayer::OutputLayer(const int & iSizeX, const int & iSizeY)
{
	mSizeX = iSizeX;
	mSizeY = iSizeY;
}


OutputLayer::~OutputLayer()
{
}

Eigen::MatrixXd OutputLayer::feedForward()
{//mock
	return Eigen::MatrixXd();
}

Eigen::MatrixXd OutputLayer::backPropagate()
{//mock
	return Eigen::MatrixXd();
}
