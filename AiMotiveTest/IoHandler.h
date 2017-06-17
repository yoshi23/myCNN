#pragma once
#include <string>
#include <vector>
#include "Dense"
class IoHandler
{
public:
	IoHandler();
	virtual ~IoHandler();

	typedef std::vector<Eigen::MatrixXd> rgbPixelMap;
	rgbPixelMap loadImage(const std::string & iFileName);
	void writePixelMapToFile(const IoHandler::rgbPixelMap & iImage);

};

