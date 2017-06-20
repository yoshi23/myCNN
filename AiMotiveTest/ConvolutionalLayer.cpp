#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include "IoHandling.h"
#include <vector>
#include <cstdlib>

#include <iostream>

ConvolutionalLayer::ConvolutionalLayer()
{
}

ConvolutionalLayer::ConvolutionalLayer(const int & iWidth, const int & iHeight, const int & iNumOfInputFeatureMaps, const int & iNumOfKernels, const int & iKernelWidth, const int & iKernelHeight, const double & iEta, const double & iEpsilon)
{
	mEta = iEta/ iNumOfInputFeatureMaps;
	mEpsilon = iEpsilon;
	mSizeX = iHeight;
	mSizeY = iWidth;

	for (int i = 0; i < iNumOfKernels; ++i)
	{
		Kernel newKernel(iNumOfInputFeatureMaps);
		for (int j = 0; j < iNumOfInputFeatureMaps; ++j)
		{
			newKernel[j] = Eigen::MatrixXd::Random(iKernelWidth, iKernelHeight);
		}
		mKernels.push_back(newKernel);
		mBias.push_back(Eigen::MatrixXd::Ones(mSizeX, mSizeY) * ((double)rand() / RAND_MAX));
	}
	mOutput.resize(iNumOfKernels);
	mGradOfActivation.resize(iNumOfKernels);
}


ConvolutionalLayer::~ConvolutionalLayer()
{
}

void ConvolutionalLayer::convolve()
{
	mGradOfActivation.resize(mOutput.size());
	Eigen::MatrixXd epsilonMat = Eigen::MatrixXd::Ones(mSizeX, mSizeY) * mEpsilon;
	std::vector<Kernel >::iterator itKernel = mKernels.begin();
	Eigen::MatrixXd wGradientOfActivation;
	Eigen::MatrixXd wNewFeatureMap;
	for (unsigned int i = 0; i < mKernels.size(); ++i)
	{
		wNewFeatureMap = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
		wGradientOfActivation = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
		for (unsigned int j = 0; j < mInput.size(); ++j)
		{
			wNewFeatureMap += convolution(mInput[j], mKernels[i][j], Layer::Valid);
		}
		wNewFeatureMap -= mBias[i];

		int wTau = 5;
		applyActivationFuncAndCalcGradient(wNewFeatureMap, wGradientOfActivation);

		mOutput[i] = wNewFeatureMap;
		mGradOfActivation[i] = wGradientOfActivation;

	}
}

void ConvolutionalLayer::feedForward(Layer * pNextLayer)
{
	convolve();
	pNextLayer->acceptInput(mOutput);
}

void ConvolutionalLayer::backPropagate(Layer * pPreviousLayer)
{
	//calculateActivationGradient(); //<--Already calculated during forward propagation
	calcDeltaOfLayer();
	weightUpdate();
	biasUpdate();

	std::vector<Eigen::MatrixXd> wWeightedDeltaOfLayer(mInput.size());

	for (unsigned int inputFeatureMaps = 0; inputFeatureMaps < mInput.size(); ++inputFeatureMaps)
	{
		wWeightedDeltaOfLayer[inputFeatureMaps] = Eigen::MatrixXd::Zero(mInput[inputFeatureMaps].rows(), mInput[inputFeatureMaps].cols());
		for (unsigned int outputFeatureMaps = 0; outputFeatureMaps < mOutput.size(); ++outputFeatureMaps)
		{
			wWeightedDeltaOfLayer[inputFeatureMaps] += convolution(mDeltaOfLayer[outputFeatureMaps], mKernels[outputFeatureMaps][inputFeatureMaps], Layer::DoubleFlip);
		}
	}

	pPreviousLayer->acceptErrorOfPrevLayer(wWeightedDeltaOfLayer);

}


void ConvolutionalLayer::acceptInput(const std::vector<Eigen::MatrixXd>& iInputMap)
{
	mInput = iInputMap;
}


void ConvolutionalLayer::acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer)
{
	mDeltaErrorOfPrevLayer = ideltaErrorOfPrevLayer;
}

void ConvolutionalLayer::calcDeltaOfLayer()
{
	mDeltaOfLayer.resize(mOutput.size());
	for (unsigned int outputFeatureMaps = 0; outputFeatureMaps < mOutput.size(); ++outputFeatureMaps)
	{
			mDeltaOfLayer[outputFeatureMaps] = mDeltaErrorOfPrevLayer[outputFeatureMaps]
				.cwiseProduct(mGradOfActivation[outputFeatureMaps]);
	}
}



void ConvolutionalLayer::weightUpdate()
{
	int wKernelWidth = mKernels[0][0].cols();
	int wKernelHeight = mKernels[0][0].rows();
	Eigen::MatrixXd d_Error_d_Weight = Eigen::MatrixXd::Zero(wKernelHeight, wKernelWidth);
	for (unsigned int numKernel = 0; numKernel < mKernels.size(); ++numKernel)
	{
		for (unsigned int kernelDepth = 0; kernelDepth < mKernels[numKernel].size(); ++kernelDepth)
		{
			for (unsigned int inputSizeX = 0; inputSizeX < mInput[kernelDepth].rows() - wKernelHeight; ++inputSizeX)
			{
				for (unsigned int inputSizeY = 0; inputSizeY < mInput[kernelDepth].cols() - wKernelWidth; ++inputSizeY)
				{
					d_Error_d_Weight += mInput[kernelDepth].block(inputSizeX, inputSizeY, wKernelHeight, wKernelWidth)
						*mDeltaOfLayer[numKernel](inputSizeX, inputSizeY);
						//*mDeltaOfLayer[kernelDepth](inputSizeX, inputSizeY);
						//.cwiseProduct( mDeltaOfLayer[kernelDepth].block(inputSizeX, inputSizeY, wKernelHeight, wKernelWidth));

				}
			}
			mKernels[numKernel][kernelDepth] += mEta * d_Error_d_Weight;
			d_Error_d_Weight = Eigen::MatrixXd::Zero(wKernelHeight, wKernelWidth);
		}
	}
}

void ConvolutionalLayer::biasUpdate()
{
	Eigen::MatrixXd d_Error_d_Bias = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
	for (unsigned int depth = 0; depth < mBias.size(); ++depth)
	{
		d_Error_d_Bias = mDeltaOfLayer[depth];
		mBias[depth].noalias() += (mEta * d_Error_d_Bias);
	}
}
