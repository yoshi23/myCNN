#pragma once
#include <string>
#include <vector>
#include "Dense"
#include "Layer.h"
#include <list>

#define IMAGE_WIDTH 52
#define IMAGE_HEIGHT 52

//A collection of functions to handle writing to/from files and console.
namespace IoHandling
{
	typedef std::vector<Eigen::MatrixXd> rgbPixelMap;
	rgbPixelMap loadImage(const std::string & iFileName);

	void saveWeightsAndBiases(const std::list<Layer*> & iLayers, const int & ID); //IMPLEMENTATION IS NOT FINISHED YET
	void loadWeightsAndBiases(Layer* iLayer); //NOT IMPLEMENTED YET

	 //writes the name of the road sign explicitly and information about the error rate.
	void nameTable(const std::vector<Eigen::MatrixXd>& iOuptut, const double& iError);
};

