#include "stdafx.h"
#include "PoolingLayer.h"
//#include "Dense"
#include <map>
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

void PoolingLayer::feedForward(Layer * pNextLayer)
{//mock
	std::map<char, Eigen::MatrixXd> fake;
	//return fake;
}

std::map<char, Eigen::MatrixXd> PoolingLayer::backPropagate()
{//mock
	std::map<char, Eigen::MatrixXd> fake;
	return fake;
}

void PoolingLayer::acceptInput(const std::map<char, Eigen::MatrixXd>&) {}