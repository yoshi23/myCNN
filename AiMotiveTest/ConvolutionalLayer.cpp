#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include "IoHandling.h"
#include <algorithm>
#include <vector>
#include <iostream>

ConvolutionalLayer::ConvolutionalLayer()
{
}

ConvolutionalLayer::ConvolutionalLayer(const int & iWidth, const int & iHeight, const int & wNumOfInputFeatureMaps, const int & iNumOfKernels, const int & iKernelWidth, const int & iKernelHeight)
{
	mSizeX = iHeight;
	mSizeY = iWidth;

	for (int i = 0; i < iNumOfKernels; ++i)
	{
		Kernel newKernel;
		newKernel.resize(wNumOfInputFeatureMaps);
		for (int j = 0; j < wNumOfInputFeatureMaps; ++j)
		{
			newKernel[j] = Eigen::MatrixXd::Random(iKernelWidth, iKernelHeight);
		}
		mKernels.push_back(newKernel);
		mBias.push_back(Eigen::MatrixXd::Ones(mSizeX, mSizeY) * Eigen::MatrixXd::Random(1, 1)(0,0));

		mOutput.push_back(Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH));
	}
}


ConvolutionalLayer::~ConvolutionalLayer()
{
}

void ConvolutionalLayer::convolve()
{
	std::vector<Kernel >::iterator itKernel = mKernels.begin();
	for (unsigned int i = 0; i < mKernels.size(); ++i)
	{
		Eigen::MatrixXd wNewFeatureMap = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
		for (unsigned int j = 0; j < mInput.size(); ++j)
		{
			wNewFeatureMap += convolution(mInput[j], mKernels[i][j], Layer::Valid);
		}
		wNewFeatureMap -= mBias[i];
		applyActivationFunction(wNewFeatureMap, 1);
		mOutput[i] = (wNewFeatureMap);
	}
}

void ConvolutionalLayer::feedForward(Layer * pNextLayer)
{
	convolve();
	pNextLayer->acceptInput(mOutput);
}

void ConvolutionalLayer::backPropagate(Layer * pPreviousLayer)
{
	calculateActivationGradient();
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

void ConvolutionalLayer::calculateActivationGradient()
{
	mGradOfActivation.resize(mOutput.size());
	Eigen::MatrixXd epsilonMat = Eigen::MatrixXd::Ones(mInput[0].rows(), mInput[0].cols()) * epsilon;

	std::vector<Kernel >::iterator itKernel = mKernels.begin();
	for (unsigned int i = 0; i < mKernels.size(); ++i)
	{
		Eigen::MatrixXd wNewFeatureMapLeft = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
		Eigen::MatrixXd wNewFeatureMapRight = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
		Eigen::MatrixXd wLeftApprox;
		Eigen::MatrixXd wRightApprox;
		for (unsigned int j = 0; j < mInput.size(); ++j)
		{
			wLeftApprox = mInput[j] - epsilonMat;
			wRightApprox = mInput[j] + epsilonMat;

			wNewFeatureMapLeft += convolution(wLeftApprox, mKernels[i][j], Layer::Valid);
			wNewFeatureMapRight += convolution(wRightApprox, mKernels[i][j], Layer::Valid);
		}
		wNewFeatureMapLeft -= mBias[i];
		wNewFeatureMapRight -= mBias[i];
		applyActivationFunction(wNewFeatureMapLeft, 1);
		applyActivationFunction(wNewFeatureMapRight, 1);
		mGradOfActivation[i] = (wNewFeatureMapRight - wNewFeatureMapLeft) / (2 * epsilon);
	}

}


void ConvolutionalLayer::calcDeltaOfLayer()
{
	mDeltaOfLayer.resize(mOutput.size());
	for (unsigned int outputFeatureMaps = 0; outputFeatureMaps < mOutput.size(); ++outputFeatureMaps)
	{
		//mDeltaOfLayer[outputFeatureMaps] = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
		//for (unsigned int inputFeatureMaps = 0; inputFeatureMaps < mInput.size(); ++inputFeatureMaps)
		//{
			//mDeltaOfLayer[outputFeatureMaps] += convolution(mDeltaErrorOfPrevLayer[outputFeatureMaps], mKernels[outputFeatureMaps][inputFeatureMaps], Layer::DoubleFlip)
			mDeltaOfLayer[outputFeatureMaps] = mDeltaErrorOfPrevLayer[outputFeatureMaps]
				.cwiseProduct(mGradOfActivation[outputFeatureMaps]);
		//}
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
						*mDeltaOfLayer[kernelDepth](inputSizeX, inputSizeY);
						//.cwiseProduct( mDeltaOfLayer[kernelDepth].block(inputSizeX, inputSizeY, wKernelHeight, wKernelWidth));

				}
			}
			mKernels[numKernel][kernelDepth] += ETA * d_Error_d_Weight;
			d_Error_d_Weight = Eigen::MatrixXd::Zero(wKernelHeight, wKernelWidth);
		}
	}
}

void ConvolutionalLayer::biasUpdate()
{
	Eigen::MatrixXd d_Error_d_Bias = Eigen::MatrixXd::Zero(mSizeX, mSizeY);
	for (int depth = 0; depth < mBias.size(); ++depth)
	{
		d_Error_d_Bias = mDeltaOfLayer[depth];
		mBias[depth] += (ETA * d_Error_d_Bias);
	}
}
