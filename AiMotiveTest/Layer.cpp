#include "stdafx.h"
#include "Layer.h"

#include <iostream>
#include <math.h>

Layer::Layer()
{
}

Layer::~Layer()
{
}

int Layer::getSizeX()
{
	return mSizeX;
}
int Layer::getSizeY()
{
	return mSizeY;
}

int Layer::getOutputDepth()
{
	return mOutput.size();
}

//Standard convolution functions implemented with 'full' and 'valid' approaches. 'Doubleflip' is just a shortcut for convolutional layers,
//Where convoltuion is performed on flipped kernels during backpropagation.
Eigen::MatrixXd Layer::convolution(const Eigen::MatrixXd &matrix, const Eigen::MatrixXd &kernel, const Layer::ConvolTypes & iType)
{
	using namespace Eigen;
	MatrixXd retMat;
	MatrixXd flippedKernel;

	const int wKernelHeight = kernel.rows();
	const int wKernelWidth = kernel.cols();
	if (iType == ConvolTypes::Valid)
	{
		flippedKernel = kernel.colwise().reverse().rowwise().reverse(); //normal rotation during 2D convolution.
		
		retMat.resize(matrix.rows() - wKernelHeight + 1, matrix.cols() - wKernelWidth + 1);
		int x(0), y(0);
		for (int i = 0; i <= matrix.rows() - wKernelHeight; ++i)
		{
			for (int j = 0; j <= matrix.cols() - wKernelWidth; j++)
			{
				retMat(x,y++) = ((matrix.block(i, j , wKernelWidth, wKernelHeight)).cwiseProduct(flippedKernel)).sum();
			}
			++x;
			y = 0;
		}
	}
	else if (iType == ConvolTypes::Full || iType == ConvolTypes::DoubleFlip)
	{
		if (iType == ConvolTypes::Full)	flippedKernel = kernel.colwise().reverse().rowwise().reverse(); //normal rotation during 2D convolution.
		else flippedKernel = kernel; //equivalent to double flipping. Neede for backprop on convolutional layers.

		MatrixXd paddedMatrix = MatrixXd::Zero(matrix.rows() + 2* wKernelHeight - 2, matrix.cols() + 2*wKernelWidth - 2);
		paddedMatrix.block(wKernelWidth - 1, wKernelHeight - 1, matrix.rows(), matrix.cols()) = matrix;
		retMat.resize(matrix.rows() + wKernelHeight - 1, matrix.cols() + wKernelWidth - 1);
		int x(0), y(0);
		for (int i = 0; i <= paddedMatrix.rows() - wKernelHeight; ++i)
		{
			for (int j = 0; j <= paddedMatrix.cols() - wKernelWidth; j++)
			{
				retMat(x, y++) = ((paddedMatrix.block(i, j, wKernelWidth, wKernelHeight)).cwiseProduct(flippedKernel)).sum();
			}
			++x;
			y = 0;
		}
	}

	return retMat;

}

void Layer::applyActivationFuncAndCalcGradient(double & iInput, double & iGradient)
{
	iInput = 1 / (1 + exp(-iInput));
	iGradient = iInput * (1 - iInput);
	//By having this analytical gradient calculation, we restrict ourselves to having iTau = 1
}

void Layer::applyActivationFuncAndCalcGradient(Eigen::MatrixXd & iInput, Eigen::MatrixXd & iGradient /*, const double & iTau*/)
{

	for (int i = 0; i < iInput.rows(); ++i)
	{
		for (int j = 0; j < iInput.cols(); ++j)
		{
			double & matrixElem = iInput(i, j);
			matrixElem = 1 / (1 + exp(-matrixElem));
			iGradient(i, j) = matrixElem * (1 - matrixElem);  //The analytical derivative of sigmoid(x).
			//By having this analytical gradient calculation, we restrict ourselves to having iTau = 1
		}
	}
}