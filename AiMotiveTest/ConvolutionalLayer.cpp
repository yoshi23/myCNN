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
		std::cout << std::endl << "hello: " << newKernel.size() << std::endl;
		for (int j = 0; j < wNumOfInputFeatureMaps; ++j)
		{
			newKernel[j] = Eigen::MatrixXd::Random(iKernelWidth, iKernelHeight);
		}
		mKernels.push_back(newKernel);
		mBias.push_back(Eigen::MatrixXd::Random(mSizeX, mSizeY));

		mOutput.push_back(Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH));
	}
}


ConvolutionalLayer::~ConvolutionalLayer()
{
}

void ConvolutionalLayer::convolve()
{
	/*for_each(mKernels.begin(), mKernels.end(), [this](Eigen::MatrixXd & iKernel)
		{
			for_each(iKernel.begin(), iKernel.end())
		Eigen::MatrixXd wNewOutput = convolution(mInput, iKernel, Layer::Valid);
		}
	);*/
		
	
	std::vector<Kernel >::iterator itKernel = mKernels.begin();
	for (int i= 0; i<mKernels.size(); ++i)//; itKernel != mKernels.end(); ++itKernel)
	{
		Eigen::MatrixXd wNewFeatureMap = Eigen::MatrixXd::Zero(mSizeX, mSizeY); 
		std::cout << "\n\nbobemobe: " << wNewFeatureMap.rows() << "x" << wNewFeatureMap.cols() << std::endl;
		for (int j = 0; j < mInput.size(); ++j)
		{
			wNewFeatureMap += convolution(mInput[j], mKernels[i][j], Layer::Valid);
			std::cout << "one convolution is done\n" << mKernels[i].size() << std::endl;
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

std::vector<Eigen::MatrixXd> ConvolutionalLayer::backPropagate()
{//MOCK
	std::vector< Eigen::MatrixXd> fake;
	return fake;
}


void ConvolutionalLayer::acceptInput(const std::vector<Eigen::MatrixXd>& iInputMap)
{
	mInput = iInputMap;
}