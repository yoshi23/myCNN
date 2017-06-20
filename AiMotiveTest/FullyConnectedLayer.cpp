#include "stdafx.h"
#include "FullyConnectedLayer.h"
#include <vector>
#include "Dense"

#include <iostream>

FullyConnectedLayer::FullyConnectedLayer()
{
}
FullyConnectedLayer::FullyConnectedLayer(
	const int & iSizeX, 
	const int & iNumOfInputFeatureMaps, 
	const int & iSizeOfPrevLayerX,
	const int & iSizeOfPrevLayerY,
	const double & iEta
)
{
	mEta = iEta; // / iNumOfInputFeatureMaps;

	mSizeX = iSizeX;
	mSizeY = 1;

	mWeights.resize(mSizeX);
	mBiases.resize(mSizeX);

	mGradOfActivation.resize(1);
	mGradOfActivation[0] = Eigen::MatrixXd::Zero(mSizeX, 1);

	mOutput.resize(1);
	mOutput[0] = (Eigen::MatrixXd::Zero(iSizeX, 1));

	for (unsigned int i = 0; i < mWeights.size(); ++i)
	{
		Weights newWeights(iNumOfInputFeatureMaps);
		Eigen::MatrixXd newBiases = Eigen::MatrixXd::Random(iNumOfInputFeatureMaps, 1);
		for (int j = 0; j < iNumOfInputFeatureMaps; ++j)
		{
			newWeights[j] = Eigen::MatrixXd::Random(iSizeOfPrevLayerX, iSizeOfPrevLayerY);
		}
		mWeights[i] = newWeights;
		mBiases[i] = static_cast<double>(Eigen::MatrixXd::Random(1, 1)(0, 0));// *iSizeOfPrevLayerX * iSizeOfPrevLayerY);
	}
}


FullyConnectedLayer::~FullyConnectedLayer()
{
}


void FullyConnectedLayer::feedForward(Layer * pNextLayer)
{
	calculateActivation();
	pNextLayer->acceptInput(mOutput);

}

void FullyConnectedLayer::backPropagate(Layer * pPreviousLayer)
{
	weightUpdate();
	biasUpdate();

	std::vector<Eigen::MatrixXd> wWeightedDeltaOfLayer(mInput.size());
	//MY_WARNING: THESE NESTED LOOPS ARE A POTENTIAL SOURCE OF PROBLEMS
	for (unsigned int neuronInPrevLayerX = 0; neuronInPrevLayerX < mInput[0].rows(); ++neuronInPrevLayerX)
	{
		//mInput[0] is a valid measurement because in the config files we can only give homogenous kernel sizes.
		for (unsigned int neuronInPrevLayerY = 0; neuronInPrevLayerY < mInput[0].cols(); ++neuronInPrevLayerY)
		{
			for (unsigned int inputFeatureMaps = 0; inputFeatureMaps < mInput.size(); ++inputFeatureMaps)
			{
				wWeightedDeltaOfLayer[inputFeatureMaps] = Eigen::MatrixXd::Zero(mInput[inputFeatureMaps].rows(), mInput[inputFeatureMaps].cols());
				for (unsigned int neuronInThisLayer = 0; neuronInThisLayer < wWeightedDeltaOfLayer.size(); ++neuronInThisLayer)
				{
					wWeightedDeltaOfLayer[inputFeatureMaps](neuronInPrevLayerX, neuronInPrevLayerY)
						+= mWeights[neuronInThisLayer][inputFeatureMaps](neuronInPrevLayerX, neuronInPrevLayerY) * mDeltaOfLayer[0](neuronInThisLayer, 0);
				
				}
			}
		}
	}
	pPreviousLayer->acceptErrorOfPrevLayer(wWeightedDeltaOfLayer);
	
}

void FullyConnectedLayer::acceptInput(const std::vector<Eigen::MatrixXd>& iInput)
{
	mInput = iInput;
}

void FullyConnectedLayer::acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer)
{
	mDeltaOfLayer.resize(mOutput.size());
	mDeltaOfLayer[0] = ideltaErrorOfPrevLayer[0].cwiseProduct(mGradOfActivation[0]);
}

void FullyConnectedLayer::calculateActivation()
{
	for (int i = 0; i < mSizeX; ++i)
	{
		double activation(0);
		double gradient(0);
		for (int j = 0; j < mInput.size(); ++j)
		{
			activation += (mInput[j].cwiseProduct(mWeights[i][j])).sum();
		}
		activation -= mBiases[i];

		applyActivationFuncAndCalcGradient(activation, gradient);

		mOutput[0](i, 0) = activation;
		mGradOfActivation[0](i, 0) = gradient;
	}
}

void FullyConnectedLayer::weightUpdate()
{
	for (unsigned int neuron = 0; neuron < mSizeX; ++neuron)
	{
		for (unsigned int inputFeatMap = 0; inputFeatMap < mWeights[neuron].size(); ++inputFeatMap)
		{
			mWeights[neuron][inputFeatMap] += (mEta * mInput[inputFeatMap] * mDeltaOfLayer[0](neuron, 0));
		}
	}
}

void FullyConnectedLayer::biasUpdate()
{
	for (int neuron = 0; neuron < mSizeX; ++neuron)
	{
		mBiases[neuron] += mEta *  mDeltaOfLayer[0](neuron, 0);
	}
}
