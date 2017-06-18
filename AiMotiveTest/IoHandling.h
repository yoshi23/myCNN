#pragma once
#include <string>
#include <vector>
#include "Dense"
#include "Layer.h"
#include <list>

#define IMAGE_WIDTH 52
#define IMAGE_HEIGHT 52

namespace IoHandling
{

	typedef std::vector<Eigen::MatrixXd> rgbPixelMap;
	rgbPixelMap loadImage(const std::string & iFileName);
	void writePixelMapToFile(const IoHandling::rgbPixelMap & iImage);

	void saveWeightsAndBiases(const std::list<Layer*> & iLayers, const int & ID);
	void loadWeightsAndBiases(Layer* iLayer);

};

