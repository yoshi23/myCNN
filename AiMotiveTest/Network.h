#pragma once
#include <list>
#include <string>
class Layer;

class Network
{
public:
	Network();
	
	~Network();

	void initialize(const std::string & networkDescriptionFile, const int & inputSizeX, const int & intputSizeY);
	void run();

private:

	std::list<Layer*> mLayers;

};

