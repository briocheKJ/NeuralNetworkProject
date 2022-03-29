#pragma once

class FeatureMap;

class Layer
{
public:
	virtual void init() = 0;
	virtual void forward() = 0;
	virtual void backward() = 0;
};