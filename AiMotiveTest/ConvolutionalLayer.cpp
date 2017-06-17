#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include <algorithm>
#include <vector>
#include <map>
#include <iostream>

ConvolutionalLayer::ConvolutionalLayer()
{
}

ConvolutionalLayer::ConvolutionalLayer(const int & iWidth, const int & iHeight, const int & wNumOfInputFeatureMaps, const int & iNumOfKernels, const int & iFilterWidth, const int & iFilterHeight)
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
			newKernel[j] = Eigen::MatrixXd::Random(iFilterWidth, iFilterHeight);
		}
		mKernels.push_back(newKernel);
		mOutput.insert(std::make_pair(i, Eigen::MatrixXd::Random(IMAGE_HEIGHT, IMAGE_WIDTH)));
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
		Eigen::MatrixXd wNewFeatureMap = convolution(mInput[0], mKernels[i][0], Layer::Valid); //First convolvation is taken out here
		//so we do not have to worry about the size of wNewFeatureMap, but set automatically by the returning value.
		std::cout << "\n\n" << wNewFeatureMap.rows() << "x" << wNewFeatureMap.cols() << std::endl;
		for (int j = 1; j < mKernels.size(); ++j)
		{
			wNewFeatureMap += convolution(mInput[j], mKernels[i][j], Layer::Valid);
		}
		wNewFeatureMap -= mBias[i];
		applyActivationFunction(wNewFeatureMap, 1);
		mOutput.insert(std::make_pair(mOutput.size()+1, wNewFeatureMap));
	}

	


}

void ConvolutionalLayer::feedForward(Layer * pNextLayer)
{
	convolve();
	pNextLayer->acceptInput(mOutput);
}

std::map<char, Eigen::MatrixXd> ConvolutionalLayer::backPropagate()
{//MOCK
	std::map<char, Eigen::MatrixXd> fake;
	return fake;
}


void ConvolutionalLayer::acceptInput(const std::map<char, Eigen::MatrixXd>& iInputMap)
{
	mInput = iInputMap;
}