#include "stdafx.h"
#include "FullyConnectedLayer.h"
#include <map>
#include <vector>
#include "Dense"

#include <iostream>

FullyConnectedLayer::FullyConnectedLayer()
{
}
FullyConnectedLayer::FullyConnectedLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY)
{
	mSizeX = iSizeX;
	mSizeY = 1;

	mWeights.resize(mSizeX);
	mBiases.resize(mSizeX);
	for (int i = 0; i < mWeights.size(); ++i)
	{
		Weights newWeights(iNumOfInputFeatureMaps);
		Eigen::MatrixXd newBiases = Eigen::MatrixXd::Random(iNumOfInputFeatureMaps, 1);
		for (int j = 0; j < iNumOfInputFeatureMaps; ++j)
		{
			newWeights[j] = Eigen::MatrixXd::Random(iSizeOfPrevLayerX, iSizeOfPrevLayerY);
		}
		mWeights[i] = newWeights;
		mBiases[i] = Eigen::MatrixXd::Random(1, 1)(0, 0);
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

void FullyConnectedLayer::backPropagate(Layer * pPreviousLayer)
{

	calculateActivationGradient();
	calcDeltaOfLayer();
	weightUpdate();
	biasUpdate();

	std::vector<Eigen::MatrixXd> wWeightedDeltaOfLayer(mInput.size());
	//MY_WARNING: THESE NESTED LOOPS ARE A POTENTIAL SOURCE OF PROBLEMS
	for (int neuronInPrevLayerX = 0; neuronInPrevLayerX < mInput[0].rows(); ++neuronInPrevLayerX)
	{
		//mInput[0] is a valid measurement because in the config files we can only give homogenous kernel sizes.
		for (int neuronInPrevLayerY = 0; neuronInPrevLayerY < mInput[0].cols(); ++neuronInPrevLayerY)
		{
			for (int inputFeatureMaps = 0; inputFeatureMaps < mInput.size(); ++inputFeatureMaps)
			{
				wWeightedDeltaOfLayer[inputFeatureMaps] = Eigen::MatrixXd::Zero(mInput[inputFeatureMaps].rows(), mInput[inputFeatureMaps].cols());
				for (int neuronInThisLayer = 0; neuronInThisLayer < wWeightedDeltaOfLayer.size(); ++neuronInThisLayer)
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
	mdeltaErrorOfPrevLayer = ideltaErrorOfPrevLayer;
}


void FullyConnectedLayer::calculateActivation()
{
	for (int i = 0; i < mSizeX; ++i)
	{
		double result(0);
		for (int j = 0; j < mInput.size(); ++j)
		{

			result += (mInput[j].cwiseProduct(mWeights[i][j])).sum();

		}
		result -= mBiases[i];
		sigmoid(result, 1);
		mOutput[0](i, 0) = result;

	}



}

void FullyConnectedLayer::calculateActivationGradient()
{
	double epsilon = 0.01;
	for (int i = 0; i < mSizeX; ++i)
	{
		double result(0);
		for (int j = 0; j < mInput.size(); ++j)
		{

			result += (mInput[j].cwiseProduct(mWeights[i][j])).sum();

		}
		result -= mBiases[i];
		double leftResult;
		double rightResult;
		leftResult -= epsilon;
		rightResult += epsilon;
		sigmoid(leftResult, 1);
		sigmoid(rightResult, 1);

		mGradOfActivation[0](i, 0) = (rightResult - leftResult) / (2 * epsilon);

	}


}



void FullyConnectedLayer::weightUpdate()
{
	Eigen::MatrixXd d_Error_d_Weight; 
	for (int neuron = 0; neuron < mWeights.size(); ++neuron)
	{
		for (int inputFeatMap = 0; inputFeatMap < mWeights[neuron].size(); ++inputFeatMap)
		{
			d_Error_d_Weight.resize(mWeights[neuron][inputFeatMap].rows(), mWeights[neuron][inputFeatMap].cols());
			d_Error_d_Weight = mInput[inputFeatMap] * mDeltaOfLayer[0](neuron, 0);

			mWeights[neuron][inputFeatMap] += ETA * d_Error_d_Weight;
		}
	}
}

void FullyConnectedLayer::biasUpdate()
{
	Eigen::MatrixXd d_Error_d_Bias = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
	for (int neuron = 0; neuron < mSizeX; ++neuron)
	{
		d_Error_d_Bias(neuron,1) = mDeltaOfLayer[0](neuron, 0);
		mBiases[neuron] += ETA * d_Error_d_Bias(neuron,1);
		
	}
}


void FullyConnectedLayer::calcDeltaOfLayer()
{

	mDeltaOfLayer[0] = mdeltaErrorOfPrevLayer[0].cwiseProduct(mGradOfActivation[0]);
}