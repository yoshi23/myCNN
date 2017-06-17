#include "stdafx.h"
#include "PoolingLayer.h"
//#include "Dense"

PoolingLayer::PoolingLayer()
{
}

PoolingLayer::PoolingLayer(const int & iSizeX, const int & iSizeY)
{
	mSizeX = iSizeX;
	mSizeY = iSizeY;
}

PoolingLayer::~PoolingLayer()
{
}

Eigen::MatrixXd PoolingLayer::feedForward()
{//mock
	return Eigen::MatrixXd();
}

Eigen::MatrixXd PoolingLayer::backPropagate()
{//mock
	return Eigen::MatrixXd();
}
