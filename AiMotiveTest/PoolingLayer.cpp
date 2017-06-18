#include "stdafx.h"
#include "PoolingLayer.h"
//#include "Dense"
#include <vector>
#include <iostream>
PoolingLayer::PoolingLayer()
{
}

PoolingLayer::PoolingLayer(
	const int & iSizeX, const int & iSizeY, 
	const Method & iMethod,
	const int & iNumOfInputFeatureMaps,
	const int & iSizeOfPrevLayerX,
	const int & iSizeOfPrevLayerY)
{
	mSizeX = iSizeOfPrevLayerX / iSizeX;
	mSizeY = iSizeOfPrevLayerY / iSizeY;
	poolingRegionX = iSizeX;
	poolingRegionY = iSizeY;
	mMethod = iMethod;

	for (int i = 0; i < iNumOfInputFeatureMaps; ++i)
	{
		mOutput.push_back(Eigen::MatrixXd::Zero(mSizeX, mSizeY));
	}

}

PoolingLayer::~PoolingLayer()
{
}

void PoolingLayer::feedForward(Layer * pNextLayer)
{
	downSample();
	pNextLayer->acceptInput(mOutput);

}

void PoolingLayer::backPropagate(Layer * pPreviousLayer)
{//mock
}

void PoolingLayer::acceptInput(const std::vector<Eigen::MatrixXd>& iInput) 
{
	mInput = iInput;
}

void PoolingLayer::downSample()
{
	for (int i = 0; i < mInput.size(); ++i)
	{
			switch(mMethod)
			{
				case Max:
				{
					for (int y = 0; y < mSizeY; ++y)
					{
						for (int x = 0; x < mSizeX; ++x)
						{
							
							//We zero-pad the input matrix, so we can comfortably iterate through and do the max-pooling, even if the
							//size of input is not a multiple of the size of region over which we want to do pooling.
							//There might be unlucky cases when this results in taht the last row or column is just a single value,
							//but this should not hinder the functioning on the system and it is the faster implementation for now.
							int remainderX = mInput[i].cols() % poolingRegionX;
							int remainderY = mInput[i].rows() % poolingRegionY;
							if (remainderX != 0 || remainderY != 0)
							{
								Eigen::MatrixXd paddedMatrix = Eigen::MatrixXd::Zero(mInput[i].rows() + remainderX, mInput[i].cols() + remainderY);
								paddedMatrix.block(0,0, mInput[i].rows(), mInput[i].cols()) = mInput[i];
								mInput[i] = paddedMatrix;
							} 		
							poolingRegionX = mInput[i].cols() / mSizeX;
							poolingRegionY = mInput[i].rows() / mSizeY;

							mOutput[i](x, y) = mInput[i].block(poolingRegionY*y, poolingRegionX*x, poolingRegionY, poolingRegionX).maxCoeff();
						}
					}
					break;
				}
				case Average:
				{//MOCK
					break;
				}
				default:
				{
					break;
				}
			}
	}

}