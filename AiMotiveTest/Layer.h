#pragma once

#include "Dense"
#include <map>

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
	virtual std::map<char, Eigen::MatrixXd> backPropagate() = 0;
	virtual void acceptInput(const std::map<char, Eigen::MatrixXd>&) = 0;

	int getSizeX();
	int getSizeY();


	Eigen::MatrixXd convolution(const Eigen::MatrixXd &matrix, const Eigen::MatrixXd &kernel, const Layer::ConvolTypes & iType);



protected:

	
	int mSizeX;
	int mSizeY;

	std::map<char, Eigen::MatrixXd> mInput;
	std::map<char, Eigen::MatrixXd> mOutput;
	//Eigen::MatrixXd mNeurons;


};

