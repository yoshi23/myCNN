#pragma once

#include "Dense"

class Layer
{
public:
	Layer();
	virtual ~Layer();

	virtual void feedForward() = 0;
	virtual void backPropagate() = 0;

	int getSizeX();
	int getSizeY();
protected:
	int mSizeX;
	int mSizeY;
private:

	Eigen::MatrixXd mNeurons;


};

