#include "stdafx.h"
#include "InputLayer.h"
//#include "Dense"

InputLayer::InputLayer()
{
}

InputLayer::InputLayer(const int & iSizeX, const int & iSizeY)
{
	mSizeX = iSizeX;
	mSizeY = iSizeY;
}

Eigen::MatrixXd InputLayer::feedForward()
{
	return mOutput;
}

Eigen::MatrixXd InputLayer::backPropagate()
{//mock
	return Eigen::MatrixXd(1, 1);
}


InputLayer::~InputLayer()
{
}
