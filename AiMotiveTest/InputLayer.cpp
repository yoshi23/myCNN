#include "stdafx.h"
#include "InputLayer.h"
#include "Layer.h"
#include "IoHandling.h"

InputLayer::InputLayer()
{
}

InputLayer::InputLayer(const int & iSizeX, const int & iSizeY)
{
	mSizeX = iSizeX;
	mSizeY = iSizeY;
	//This is just to have some placeholder data there and more importantly to provide meaningful number about the outgoing
	//featuremaps for the next (1st hidden) layer.
	mOutput.push_back(Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH));
	mOutput.push_back(Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH));
	mOutput.push_back(Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH));
}

void InputLayer::feedForward(Layer * pNextLayer)
{
	pNextLayer->acceptInput(mInput);
}
void InputLayer::backPropagate(Layer * pPreviousLayer)
{//mock

}


void InputLayer::acceptInput(const std::vector<Eigen::MatrixXd>& iImage)
{
	mInput = mOutput = iImage;
}


InputLayer::~InputLayer()
{
}

void InputLayer::acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer)
{
}