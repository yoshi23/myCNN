#pragma once

#include "Dense"
#include <vector>



class Layer
{
public:

	enum ConvolTypes
	{
		Full,
		//Same, <--This type is not needed.
		Valid, 
		DoubleFlip //this is not a standard convolution type. 
		//I just want to explit for performance that for the 
		//convolution layer, during backpropagation we use 
		//the flipped kernel, so for the convolution 
		//we would flip twice, instead I just don't flip.
	};

	Layer();
	virtual ~Layer();

	virtual void feedForward(Layer * pNextLayer) = 0;
	virtual void backPropagate(Layer * pPreviousLayer) = 0;
	virtual void acceptInput(const std::vector<Eigen::MatrixXd>&) = 0;
	virtual void acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>&) = 0;

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
	std::vector<Eigen::MatrixXd> mGradOfActivation;
	std::vector<Eigen::MatrixXd> mDeltaOfLayer;
	std::vector<Eigen::MatrixXd> mDeltaErrorOfPrevLayer;

	
	const double ETA = -10; // -0.4;
	const double epsilon = 0.01; //for calculating activation gradients
};

