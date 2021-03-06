#pragma once
#include<fstream>

class FeatureMap;

class Layer
{
public:
	virtual ~Layer(){}
	virtual void init(ifstream&) = 0;
	virtual void forward(double (*active)(double)) = 0;
	virtual void backward(double (*activegrad)(double)) = 0;
	virtual void update(double alpha) = 0;
	virtual void randomize() = 0;
};