#pragma once
#include "FullyConnectedLayer.h"
#include "Dense"
class OutputLayer :
	public FullyConnectedLayer
{
public:
	OutputLayer();
	OutputLayer(const int & iSizeX, const int & iNumOfInputFeatureMaps, const int & iSizeOfPrevLayerX, const int & iSizeOfPrevLayerY);
	~OutputLayer();

	void feedForward(const Eigen::MatrixXd & iExpectedOutput);
	void backPropagate(Layer * pPreviousLayer, const Eigen::MatrixXd & iExpectedOutput);
	void acceptErrorOfPrevLayer(const std::vector<Eigen::MatrixXd>& ideltaErrorOfPrevLayer);

	double getOutputError();

private:
	double mError;
	Eigen::MatrixXd mD_Error_d_Activation;
	void provideOutput();

	//Calculating standard error function
	void calculateError(const Eigen::MatrixXd & iExpectedOutput);

	//Calculating the derivative of error as a function of activation
	void calc_d_Error_d_Activation(const Eigen::MatrixXd & iExpectedOutput);

	//Calculates delta error of layer.
	void calcDeltaOfLayer();
	
	//void weightUpdate();
	
};

