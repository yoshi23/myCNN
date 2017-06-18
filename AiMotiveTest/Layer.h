#pragma once

#include "Dense"
#include <vector>

#define IMAGE_WIDTH 52
#define IMAGE_HEIGHT 52

class Layer
{
public:

	enum ConvolTypes
	{
		Full,
		Same,
		Valid
	};

	Layer();
	virtual ~Layer();

	virtual void feedForward(Layer * pNextLayer) = 0;
	virtual std::vector<Eigen::MatrixXd> backPropagate() = 0;
	virtual void acceptInput(const std::vector<Eigen::MatrixXd>&) = 0;

	int getSizeX();
	int getSizeY();
	int getOutPutSize();

	Eigen::MatrixXd convolution(const Eigen::MatrixXd &matrix, const Eigen::MatrixXd &kernel, const Layer::ConvolTypes & iType);

	void applyActivationFunction(Eigen::MatrixXd &matrix, const double & iTau);
	void sigmoid(double & iVal, const double & iTau);

protected:

	int mSizeX;
	int mSizeY;

	std::vector<Eigen::MatrixXd> mInput;
	std::vector<Eigen::MatrixXd> mOutput;

};

