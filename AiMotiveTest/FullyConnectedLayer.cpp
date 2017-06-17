#include "stdafx.h"
#include "FullyConnectedLayer.h"
#include <map>
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


void FullyConnectedLayer::feedForward(Layer * pNextLayer)
{//mock
	std::map<char, Eigen::MatrixXd> fake;
	//return fake;
}

std::map<char, Eigen::MatrixXd> FullyConnectedLayer::backPropagate()
{//mock
	std::map<char, Eigen::MatrixXd> fake;
	return fake;
}

void FullyConnectedLayer::acceptInput(const std::map<char, Eigen::MatrixXd>&) {}