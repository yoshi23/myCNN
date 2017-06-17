#include "stdafx.h"
#include "PoolingLayer.h"
//#include "Dense"
#include <vector>
#include <iostream>
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

}

std::vector<Eigen::MatrixXd> PoolingLayer::backPropagate()
{//mock
	std::vector<Eigen::MatrixXd> fake;
	return fake;
}

void PoolingLayer::acceptInput(const std::vector<Eigen::MatrixXd>&) 
{
	std::cout << "\nPooling layer accepts input \n";
}