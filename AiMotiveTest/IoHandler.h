#pragma once
#include <string>
class IoHandler
{
public:
	IoHandler();
	virtual ~IoHandler();

	bool loadImage(const std::string iFileName);
};

