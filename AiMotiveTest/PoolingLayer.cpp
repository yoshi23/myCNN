#include "stdafx.h"
#include "PoolingLayer.h"
#include <vector>
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
	mPoolingRegionX = iSizeX;
	mPoolingRegionY = iSizeY;
	mMethod = iMethod;
	mEta = 0; //unused for pooling

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
{
	//delta error has been already calculated during feed forward phase.
	//For a real system, it should be calculated seperately, because it slows down the system unnecessarily if we do not perform learning.
	for (unsigned int inputMaps = 0; inputMaps < mInput.size(); ++inputMaps)
	{
		for (unsigned int x = 0; x < mWeightedDeltaOfLayer[inputMaps].cols(); ++x)
		{
			for (unsigned int y = 0; y < mWeightedDeltaOfLayer[inputMaps].rows(); ++y)
			{
				if (mWeightedDeltaOfLayer[inputMaps](x,y) == 1)
				{
					mWeightedDeltaOfLayer[inputMaps](x, y) = mDeltaErrorOfPrevLayer[inputMaps](x / mPoolingRegionX, y / mPoolingRegionY);
				}
			}
		}
		
	}
	pPreviousLayer->acceptErrorOfPrevLayer(mWeightedDeltaOfLayer);
}

void PoolingLayer::acceptInput(const std::vector<Eigen::MatrixXd>& iInput) 
{
	mInput = iInput;
}

void PoolingLayer::acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer)
{
	mDeltaErrorOfPrevLayer = ideltaErrorOfPrevLayer;
}

void PoolingLayer::downSample()
{
	mWeightedDeltaOfLayer.resize(mInput.size());
	Eigen::MatrixXd wSubRegion;
	for (int i = 0; i < mInput.size(); ++i)
	{
			mWeightedDeltaOfLayer[i] = Eigen::MatrixXd::Zero(mInput[i].rows(), mInput[i].cols());

			switch(mMethod)
			{
				case Max:
				{
					//We zero-pad the input matrix, so we can comfortably iterate through and do the max-pooling, even if the
					//size of input is not a multiple of the size of region over which we want to do pooling.
					//There might be unlucky cases when this results in taht the last row or column is just a single value,
					//but this should not hinder the functioning on the system and it is the faster implementation for now.
					int remainderX = mInput[i].cols() % mPoolingRegionX;
					int remainderY = mInput[i].rows() % mPoolingRegionY;
					if (remainderX != 0 || remainderY != 0)
					{
						Eigen::MatrixXd paddedMatrix = Eigen::MatrixXd::Zero(mInput[i].rows() + remainderX, mInput[i].cols() + remainderY);
						paddedMatrix.block(0, 0, mInput[i].rows(), mInput[i].cols()) = mInput[i];
						mInput[i] = paddedMatrix;
					}
					mPoolingRegionX = mInput[i].cols() / mSizeX;
					mPoolingRegionY = mInput[i].rows() / mSizeY;

					for (int y = 0; y < mSizeY; ++y)
					{
						for (int x = 0; x < mSizeX; ++x)
						{			
							wSubRegion = mInput[i].block(mPoolingRegionY*y, mPoolingRegionX*x, mPoolingRegionY, mPoolingRegionX);
							mOutput[i](x, y) = wSubRegion.maxCoeff();

							for (int subX = 0; subX < mPoolingRegionX; ++subX)
							{
								for (int subY = 0; subY < mPoolingRegionY; ++subY)
								{
									if (wSubRegion.maxCoeff() == wSubRegion(subY, subX))
									{
										mWeightedDeltaOfLayer[i](mPoolingRegionY*y + subY, mPoolingRegionX*x + subX) = 1; //later, during backpropagation we will multiply this matrix by the delta of prev layer. 
										//This way, only the max element will have non-zero value, and will forward the error through the layer.
									}
								}
							}
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


