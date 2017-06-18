#pragma once
#include <list>
#include <string>
class Layer;

class Network
{
public:

	enum RunningMode
	{
		Learning,
		Working
	};

	Network();
	
	~Network();

	bool isLearning();

	void initialize(const std::string & iNetworkDescriptionFile, const int & inputSizeX, const int & intputSizeY);
	int run(const std::string & iDirectory, const int & dirNum);

private:

	std::list<Layer*> mLayers;

	RunningMode mRunningMode;

};

