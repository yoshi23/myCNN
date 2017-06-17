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
}

void InputLayer::feedForward(Layer * pNextLayer)
{
	pNextLayer->acceptInput(mInputImage);
}

std::map<char, Eigen::MatrixXd> InputLayer::backPropagate()
{//mock
	std::map<char, Eigen::MatrixXd> fake;
	return fake;
}

void InputLayer::acceptInput(const IoHandler::rgbPixelMap & iImage)
{
	mInput = iImage;
}


InputLayer::~InputLayer()
{
}
