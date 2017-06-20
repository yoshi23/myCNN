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
	//Inverse pooling has to be done.
	//Pooling does not introduce error.

	std::vector<Eigen::MatrixXd> wWeightedDeltaOfLayer(mInput.size());
	//MY_WARNING: THESE NESTED LOOPS ARE A POTENTIAL SOURCE OF PROBLEMS
	for (int neuronInPrevLayerX = 0; neuronInPrevLayerX < mInput[0].rows() - mPoolingRegionX; ++neuronInPrevLayerX)
	{
		//mInput[0] is a valid measurement because in the config files we can only give homogenous kernel sizes.
		for (int neuronInPrevLayerY = 0; neuronInPrevLayerY < mInput[0].cols() - mPoolingRegionY; ++neuronInPrevLayerY)
		{
			for (int inputFeatureMaps = 0; inputFeatureMaps < mInput.size(); ++inputFeatureMaps)
			{
				wWeightedDeltaOfLayer[inputFeatureMaps] = Eigen::MatrixXd::Zero(mInput[inputFeatureMaps].rows(), mInput[inputFeatureMaps].cols());
				for (int neuronInThisLayer = 0; neuronInThisLayer < wWeightedDeltaOfLayer.size(); ++neuronInThisLayer)
				{
					for (int x = 0; x < mPoolingRegionX; ++x)
					{
						for (int y = 0; y < mPoolingRegionY; ++y)
						{
							Eigen::MatrixXd wSubRegion = mInput[inputFeatureMaps].block(mPoolingRegionY*y, mPoolingRegionX*x, mPoolingRegionY, mPoolingRegionX);
							if (wSubRegion.maxCoeff() == wSubRegion(x,y))
							{
								wWeightedDeltaOfLayer[inputFeatureMaps](neuronInPrevLayerY + y, neuronInPrevLayerX + x) = wSubRegion(x, y);
							}
							
						}
					}
				}
			}
		}
	}
	pPreviousLayer->acceptErrorOfPrevLayer(wWeightedDeltaOfLayer);
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
	//std::vector<Eigen::MatrixXd> wWeightedDeltaOfLayer(mInput.size());

	for (int i = 0; i < mInput.size(); ++i)
	{
			//wWeightedDeltaOfLayer[inputFeatureMaps] = Eigen::MatrixXd::Zero(mInput[inputFeatureMaps].rows(), mInput[inputFeatureMaps].cols());

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
							mOutput[i](x, y) = mInput[i].block(mPoolingRegionY*y, mPoolingRegionX*x, mPoolingRegionY, mPoolingRegionX).maxCoeff();
						}
					}
					break;
				}
				case Average:
				{
					//MOCK
					break;
				}
				default:
				{
					break;
				}
			}
	}

}


