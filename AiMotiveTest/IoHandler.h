#pragma once
#include <string>
#include <map>
#include "Dense"
class IoHandler
{
public:
	IoHandler();
	virtual ~IoHandler();

	typedef std::map<char, Eigen::MatrixXd> rgbPixelMap;
	rgbPixelMap loadImage(const std::string & iFileName);
	void writePixelMapToFile(const IoHandler::rgbPixelMap & iImage);

};

