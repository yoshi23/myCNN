#include "stdafx.h"
#include "OutputLayer.h"
#include "IoHandling.h"
#include <vector>

OutputLayer::OutputLayer()
{
}

OutputLayer::OutputLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY, const double & iEta, const double & iEpsilon)
	: FullyConnectedLayer(iSizeX, iNumOfInputFeatureMaps, iSizeOfPrevLayerX,  iSizeOfPrevLayerY,iEta, iEpsilon)
{
	mError = 0;
	mD_Error_d_Activation = Eigen::MatrixXd::Zero(iSizeX, 1);

	mDeltaOfLayer.resize(1);
	mDeltaOfLayer[0] = Eigen::MatrixXd::Zero(iSizeX, 1);
}

OutputLayer::~OutputLayer()
{
}

void OutputLayer::provideOutput(const Eigen::MatrixXd & iExpectedOutput)
{
	IoHandling::nameTable(mOutput,mError, iExpectedOutput);
}

void OutputLayer::feedForward(const Eigen::MatrixXd & iExpectedOutput)
{
	calculateActivation();
	calculateError(iExpectedOutput);
	provideOutput(iExpectedOutput);
}

void OutputLayer::backPropagate(Layer * pPreviousLayer, const Eigen::MatrixXd & iExpectedOutput)
{
	calculateError(iExpectedOutput);
	calc_d_Error_d_Activation(iExpectedOutput);
	calcDeltaOfLayer();
	weightUpdate();
	biasUpdate();

	std::vector<Eigen::MatrixXd> wWeightedDeltaOfLayer(mInput.size()); 

	//WARNING: THESE NESTED LOOPS SHOULD BE RECONSIDERED ARE A POTENTIAL SOURCE OF PROBLEMS
	for (unsigned int neuronInPrevLayerX = 0; neuronInPrevLayerX < mInput[0].rows(); ++neuronInPrevLayerX)
	{
		for (unsigned int neuronInPrevLayerY = 0; neuronInPrevLayerY < mInput[0].cols(); ++neuronInPrevLayerY)
		{
			for (unsigned int inputFeatureMaps = 0; inputFeatureMaps < mInput.size(); ++inputFeatureMaps)
			{
				wWeightedDeltaOfLayer[inputFeatureMaps] = Eigen::MatrixXd::Zero(mInput[inputFeatureMaps].rows(), mInput[inputFeatureMaps].cols());
				for (unsigned int neuronInThisLayer = 0; neuronInThisLayer < wWeightedDeltaOfLayer.size(); ++neuronInThisLayer)
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


#include <iostream>

//Calculating the derivative of error as a function of activation
void OutputLayer::calc_d_Error_d_Activation(const Eigen::MatrixXd & iExpectedOutput)
{
		mD_Error_d_Activation = mOutput[0] - iExpectedOutput;
		
}

void OutputLayer::calcDeltaOfLayer()
{
	mDeltaOfLayer[0] = mD_Error_d_Activation.cwiseProduct(mGradOfActivation[0]);
	std::cout << "error of activation: \n" << mD_Error_d_Activation << "\ngrad of actiovation: \n"<< mGradOfActivation[0]<<std::endl;
}

void OutputLayer::acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer)
{
	mDeltaErrorOfPrevLayer = ideltaErrorOfPrevLayer;
}

double OutputLayer::getOutputError()
{
	return mError;
}