#pragma once

#include "Dense"

class Layer
{
public:
	Layer();
	virtual ~Layer();

	virtual Eigen::MatrixXd feedForward() = 0;
	virtual Eigen::MatrixXd backPropagate() = 0;

	int getSizeX();
	int getSizeY();
protected:
	int mSizeX;
	int mSizeY;

	Eigen::MatrixXd mInput;
	Eigen::MatrixXd mOutput;
	//Eigen::MatrixXd mNeurons;


};

