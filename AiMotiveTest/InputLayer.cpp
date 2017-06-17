#include "stdafx.h"
#include "InputLayer.h"
#include "Layer.h"
//#include "Dense"
#include <map>
InputLayer::InputLayer()
{
}

InputLayer::InputLayer(const int & iSizeX, const int & iSizeY)
{
	mSizeX = iSizeX;
	mSizeY = iSizeY;
	//This is just to have some placeholder data there and more importantly to provide meaningful number about the outgoing
	//featuremaps for the next (1st hidden) layer.
	mOutput.insert(std::make_pair('r', Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH)));
	mOutput.insert(std::make_pair('g', Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH)));
	mOutput.insert(std::make_pair('b', Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH)));
}

void InputLayer::feedForward(Layer * pNextLayer)
{
	pNextLayer->acceptInput(mInput);
}

std::map<char, Eigen::MatrixXd> InputLayer::backPropagate()
{//mock
	std::map<char, Eigen::MatrixXd> fake;
	return fake;
}


void InputLayer::acceptInput(const std::map<char, Eigen::MatrixXd>& iImage)
{
	mInput = mOutput = iImage;
}


InputLayer::~InputLayer()
{
}
