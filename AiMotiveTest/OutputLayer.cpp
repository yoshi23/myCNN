#include "stdafx.h"
#include "OutputLayer.h"
#include <iostream>
#include <map>
#include <vector>

OutputLayer::OutputLayer()
{
}

OutputLayer::OutputLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY)
	: FullyConnectedLayer(iSizeX, iNumOfInputFeatureMaps, iSizeOfPrevLayerX,  iSizeOfPrevLayerY)
{
	mError = 0;
	mD_Error_d_Activation = Eigen::MatrixXd::Zero(iSizeX, 1);

	mDeltaOfLayer.resize(1);
	mDeltaOfLayer[0] = Eigen::MatrixXd::Zero(iSizeX, 1);
}

OutputLayer::~OutputLayer()
{
}

void OutputLayer::provideOutput()
{
	std::cout << "Network forward propagation has reached the end of the network! Hurray!\n";
	std::cout << mOutput[0];
}

void OutputLayer::feedForward()
{
	calculateActivation();
	provideOutput();
}

void OutputLayer::backPropagate(Layer * pPreviousLayer, const Eigen::MatrixXd & iExpectedOutput)
{
	calculateError(iExpectedOutput);
	calc_d_Error_d_Activation(iExpectedOutput);
	calculateActivationGradient();
	calcDeltaOfLayer();
	weightUpdate();
	biasUpdate();


	std::vector<Eigen::MatrixXd> wWeightedDeltaOfLayer(mInput.size()); 
	//MY_WARNING: THESE NESTED LOOPS ARE A POTENTIAL SOURCE OF PROBLEMS
	for (int neuronInPrevLayerX = 0; neuronInPrevLayerX < mInput[0].rows(); ++neuronInPrevLayerX)
	{
		for (int neuronInPrevLayerY = 0; neuronInPrevLayerY < mInput[0].cols(); ++neuronInPrevLayerY)
		{
			for (int inputFeatureMaps = 0; inputFeatureMaps < mInput.size(); ++inputFeatureMaps)
			{
				wWeightedDeltaOfLayer[inputFeatureMaps] = Eigen::MatrixXd::Zero(mInput[inputFeatureMaps].rows(), mInput[inputFeatureMaps].cols());
				for (int neuronInThisLayer = 0; neuronInThisLayer < wWeightedDeltaOfLayer.size(); ++neuronInThisLayer)
				{
					wWeightedDeltaOfLayer[inputFeatureMaps](neuronInPrevLayerX, neuronInPrevLayerY) 
						+= mWeights[neuronInThisLayer][inputFeatureMaps](neuronInPrevLayerX, neuronInPrevLayerY) * mDeltaOfLayer[0](neuronInThisLayer,0);
				}
			}
		}
	}
	pPreviousLayer->acceptErrorOfPrevLayer(wWeightedDeltaOfLayer);

}

//Calculating standard error function
void OutputLayer::calculateError(const Eigen::MatrixXd & iExpectedOutput)
{
	for(int i=0; i < mOutput[0].rows(); ++i)
	{
		mError += pow(mOutput[0](i, 0) - iExpectedOutput(i, 0), 2);
	}
	mError /= (2 * mOutput[0].rows());
}

void OutputLayer::calc_d_Error_d_Activation(const Eigen::MatrixXd & iExpectedOutput)
{
	for (int i = 0; i < mOutput[0].rows(); ++i)
	{
		mD_Error_d_Activation(i,0) = mOutput[0](i, 0) - iExpectedOutput(i, 0);
	}
}

void OutputLayer::calcDeltaOfLayer()
{
/*	for (int i = 0; i < mOutput[0].rows(); ++i)
	{
		mD_Error_d_Activation(i, 0) = mOutput[0](i, 0) - iExpectedOutput(i, 0);
	}*/
	mDeltaOfLayer[0] = mD_Error_d_Activation.cwiseProduct(mGradOfActivation[0]);
}

/*void OutputLayer::weightUpdate()
{
	std::vector<Weights > d_Error_d_Weight(mWeights.size());
	for (int i = 0; i < mWeights.size(); ++i)
	{

	}

}*/


void OutputLayer::acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer)
{
	mdeltaErrorOfPrevLayer = ideltaErrorOfPrevLayer;
}