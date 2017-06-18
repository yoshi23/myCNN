#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include <algorithm>
#include <vector>
#include <map>
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
	mdeltaErrorOfPrevLayer = ideltaErrorOfPrevLayer;
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
		for (unsigned int j = 0; j < mInput.size(); ++j)
		{
			wNewFeatureMapLeft = mInput[j] - epsilonMat;
			wNewFeatureMapRight = mInput[j] + epsilonMat;

			wNewFeatureMapLeft += convolution(wNewFeatureMapLeft, mKernels[i][j], Layer::Valid);
			wNewFeatureMapRight += convolution(wNewFeatureMapRight, mKernels[i][j], Layer::Valid);
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
		for (unsigned int inputFeatureMaps = 0; inputFeatureMaps < mInput.size(); ++inputFeatureMaps)
		{
			mDeltaOfLayer[outputFeatureMaps] += convolution(mdeltaErrorOfPrevLayer[outputFeatureMaps], mKernels[outputFeatureMaps][inputFeatureMaps], Layer::DoubleFlip)
				.cwiseProduct(mGradOfActivation[outputFeatureMaps]);
		}
	}
}



void ConvolutionalLayer::weightUpdate()
{
	//std::vector<Eigen::MatrixXd> d_Error_d_Weight; 
	//Kernels have the same size currently, so the [0][0] reference is valid for initiating for all sizes.
	Eigen::MatrixXd d_Error_d_Weight = Eigen::MatrixXd::Zero(mKernels[0][0].rows(), mKernels[0][0].cols());

	for (unsigned int numKernel = 0; numKernel < mKernels.size(); ++numKernel)
	{
		//d_Error_d_Weight.resize(mKernels[numKernel].size());
		for (unsigned int kernelDepth = 0; kernelDepth < mKernels.size(); ++kernelDepth)
		{


			//d_Error_d_Weight += convolution()


			//mInput[kernelDepth]()
		//	for (unsigned int i = 0; i < d_Error_d_Weight.rows(); ++i)
		//	{
		//		for (unsigned int j = 0; j < d_Error_d_Weight.cols(); ++j)
		//		{
					for (unsigned int inputSizeX = 0; inputSizeX < mInput[kernelDepth].rows(); ++inputSizeX)
					{
						for (unsigned int inputSizeY = 0; inputSizeY < mInput[kernelDepth].cols(); ++inputSizeY)
						{
							d_Error_d_Weight += mInput[kernelDepth].block(inputSizeX, inputSizeY, mKernels[0][0].rows(), mKernels[0][0].cols())
								.cwiseProduct( mDeltaOfLayer[kernelDepth]);

						}
					}
					
		//		}
		//	}
			mKernels[numKernel][kernelDepth] += ETA * d_Error_d_Weight;
			d_Error_d_Weight = Eigen::MatrixXd::Zero(mKernels[0][0].rows(), mKernels[0][0].cols());
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
