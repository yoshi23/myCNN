#include "stdafx.h"
#include "FullyConnectedLayer.h"

#include "Dense"
FullyConnectedLayer::FullyConnectedLayer()
{
}
FullyConnectedLayer::FullyConnectedLayer(const int & iSizeX, const int & iSizeY)
{
	mSizeX = iSizeX;
	mSizeY = iSizeY;
}


FullyConnectedLayer::~FullyConnectedLayer()
{
}


Eigen::MatrixXd FullyConnectedLayer::feedForward()
{//mock
	return Eigen::MatrixXd(1, 1);
}

Eigen::MatrixXd FullyConnectedLayer::backPropagate()
{//mock
	return Eigen::MatrixXd(1, 1);
}
