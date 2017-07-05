#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include "IoHandling.h"
#include <vector>
#include <cstdlib>

#include <iostream>

ConvolutionalLayer::ConvolutionalLayer()
{
}

ConvolutionalLayer::ConvolutionalLayer(const int & iWidth, const int & iHeight, const int & iNumOfInputFeatureMaps, const int & iNumOfKernels, const int & iKernelWidth, const int & iKernelHeight, const double & iEta)
{
	mEta = iEta;
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
		mBias.push_back(Eigen::MatrixXd::Ones(mSizeX, mSizeY) * (double)rand() / RAND_MAX);
	}
	mOutput.resize(iNumOfKernels);
	mGradOfActivation.resize(iNumOfKernels);
}


ConvolutionalLayer::~ConvolutionalLayer()
{
}



void ConvolutionalLayer::feedForward(Layer * pNextLayer)
{
	convolve();
	pNextLayer->acceptInput(mOutput);
}

void ConvolutionalLayer::backPropagate(Layer * pPreviousLayer)
{
	//Algorithms are based on derivations found here:
	//http://jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
	//https://grzegorzgwardys.wordpress.com/2016/04/22/8/
	//https://github.com/integeruser/MNIST-cnn

	//weightUpdate();
	//biasUpdate();

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

	mDeltaOfLayer.resize(mOutput.size());
	for (unsigned int outputFeatureMaps = 0; outputFeatureMaps < mOutput.size(); ++outputFeatureMaps)
	{
		mDeltaOfLayer[outputFeatureMaps] = ideltaErrorOfPrevLayer[outputFeatureMaps]
			.cwiseProduct(mGradOfActivation[outputFeatureMaps]);
	}
}

void ConvolutionalLayer::convolve()
{
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
		applyActivationFuncAndCalcGradient(wNewFeatureMap, wGradientOfActivation);
		mOutput[i] = wNewFeatureMap;
		mGradOfActivation[i] = wGradientOfActivation;
	}
}


void ConvolutionalLayer::weightUpdate()
{
	int wKernelWidth = mKernels[0][0].cols();
	int wKernelHeight = mKernels[0][0].rows();
	double wD_Error_d_Weight(0);
	for (unsigned int numKernel = 0; numKernel < mKernels.size(); ++numKernel)
	{
		for (unsigned int kernelDepth = 0; kernelDepth < mKernels[numKernel].size(); ++kernelDepth)
		{
			for (int x = 0; x < wKernelHeight; ++x)
			{
				for (int y = 0; y < wKernelWidth; ++y)
				{
					Eigen::MatrixXd wPrevActiveWindow = mInput[kernelDepth].block(x, y, mSizeX - wKernelHeight, mSizeY - wKernelWidth);
					Eigen::MatrixXd wDeltaWindow = mDeltaOfLayer[numKernel].block(x, y, mSizeX - wKernelHeight, mSizeY - wKernelWidth);

					wD_Error_d_Weight = (wPrevActiveWindow.cwiseProduct(wDeltaWindow)).sum();
					mKernels[numKernel][kernelDepth](x, y) += mEta * wD_Error_d_Weight;
				}
			}
		}
	}
}

void ConvolutionalLayer::biasUpdate()
{
	Eigen::MatrixXd d_Error_d_Bias = Eigen::MatrixXd::Ones(mSizeX, mSizeY);
	for (unsigned int depth = 0; depth < mBias.size(); ++depth)
	{
		d_Error_d_Bias *= (mDeltaOfLayer[depth].sum());
		mBias[depth].noalias() += (mEta * d_Error_d_Bias);
	}
}
