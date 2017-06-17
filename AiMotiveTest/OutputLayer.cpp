#include "stdafx.h"
#include "OutputLayer.h"
//#include "Dense"
#include <map>
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

std::map<char, Eigen::MatrixXd> OutputLayer::backPropagate()
{//mock
	std::map<char, Eigen::MatrixXd> fake;
	return fake;
}

void OutputLayer::acceptInput(const std::map<char, Eigen::MatrixXd>&)
{
}

