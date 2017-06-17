#include "stdafx.h"
#include "FullyConnectedLayer.h"
#include <map>
#include <vector>
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

}

std::vector<Eigen::MatrixXd> FullyConnectedLayer::backPropagate()
{//mock
	std::vector<Eigen::MatrixXd> fake;
	return fake;
}

void FullyConnectedLayer::acceptInput(const std::vector<Eigen::MatrixXd>&) {}