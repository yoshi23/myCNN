#pragma once

#include "Dense"
#include <vector>


//All other layertypes inherit from this. Abstract class.
class Layer
{
public:

	enum ConvolTypes
	{
		Full,
		Valid, 
		DoubleFlip //this is not a standard convolution type. 
		//In the convolution layer, during backpropagation we use 
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
	int getOutputDepth();
	
	Eigen::MatrixXd convolution(const Eigen::MatrixXd &matrix, const Eigen::MatrixXd &kernel, const Layer::ConvolTypes & iType);


	void applyActivationFuncAndCalcGradient(double & iInput,double &iGradient);
	void applyActivationFuncAndCalcGradient(Eigen::MatrixXd & iInput, Eigen::MatrixXd &iGradient); 
	//We switched to analytical calculation of the gradient of Sigmoid ( = s*(1-s)), but this restricts us to Tau = 1 at the moment.

protected:

	int mSizeX;
	int mSizeY;

	std::vector<Eigen::MatrixXd> mInput;
	std::vector<Eigen::MatrixXd> mOutput;
	std::vector<Eigen::MatrixXd> mGradOfActivation;
	std::vector<Eigen::MatrixXd> mDeltaOfLayer;
	std::vector<Eigen::MatrixXd> mDeltaErrorOfPrevLayer;
	
	double mEta; //For learning rate. Read from config file;

};

