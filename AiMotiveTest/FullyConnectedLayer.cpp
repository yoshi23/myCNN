#include "stdafx.h"
#include "FullyConnectedLayer.h"
#include <map>
#include <vector>
#include "Dense"
FullyConnectedLayer::FullyConnectedLayer()
{
}
FullyConnectedLayer::FullyConnectedLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY)
{
	mSizeX = iSizeX;
	mSizeY = 1;

	mWeights.resize(mSizeX);
	for (int i = 0; i < mWeights.size(); ++i)
	{
		Weights newWeights(iNumOfInputFeatureMaps);
		for (int j = 0; j < iNumOfInputFeatureMaps; ++j)
		{
			newWeights[j] = Eigen::MatrixXd::Random(iSizeOfPrevLayerX, iSizeOfPrevLayerY);
		}
		mWeights[i] = newWeights;
	}

	mOutput.resize(0);
	mOutput.push_back(Eigen::MatrixXd::Zero(iSizeX, 1));
}


FullyConnectedLayer::~FullyConnectedLayer()
{
}


void FullyConnectedLayer::feedForward(Layer * pNextLayer)
{
	calculateActivation();
	pNextLayer->acceptInput(mOutput);

}

std::vector<Eigen::MatrixXd> FullyConnectedLayer::backPropagate()
{//mock
	std::vector<Eigen::MatrixXd> fake;
	return fake;
}

void FullyConnectedLayer::acceptInput(const std::vector<Eigen::MatrixXd>& iInput)
{
	mInput = iInput;
}

void FullyConnectedLayer::calculateActivation()
{
	for (int i = 0; i < mSizeX; ++i)
	{
		for (int j = 0; j < mInput.size(); ++j)
		{
			double result(0);
			//convolution(mInput[j], mWeights[i][j], Layer::Valid);
			result = (mInput[j].cwiseProduct(mWeights[i][j])).sum();
			sigmoid(result, 1);
			mOutput[0](i, 0) = result;
		}

	}



}

