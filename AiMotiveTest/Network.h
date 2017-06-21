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


	//Builds and configures the network based on file name iConfiguration.
	void build(const int & iConfiguration);
	//Starts the system running on the image samples in iDirectory.
	int run(const std::string & iDirectory);

private:

	//This container is the actual network.
	std::list<Layer*> mLayers;

	int mConfiguration;
	RunningMode mRunningMode;

};

