#include "stdafx.h"
#include "OutputLayer.h"
//#include "Dense"
#include <iostream>
#include <map>
#include <vector>
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

void OutputLayer::feedForward(Layer * pNextLayer)
{//mock
}

std::vector< Eigen::MatrixXd> OutputLayer::backPropagate()
{//mock
	std::vector< Eigen::MatrixXd> fake;
	return fake;
}

void OutputLayer::acceptInput(const std::vector<Eigen::MatrixXd>&)
{
}

void OutputLayer::provideOutput()
{
	std::cout << "Network forward propagation has reached the end of the network! Hurray!\n";
}

