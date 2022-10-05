#pragma once
#include <fstream>
class FeatureMap;

const int THREAD_NUM=16; //线程数

class Layer
{
public:
	virtual ~Layer() {}
	virtual void init(std::ifstream&) = 0;
	virtual void forward(int pid, double (*active)(double)) = 0; //pid：“线程编号”
	virtual void backward(int pid, double (*activegrad)(double)) = 0; //pid：“线程编号”
	virtual void update(double alpha) = 0; //用Buffer更新可训练参数，学习率alpha
	virtual void updateBuffer(int pid) = 0; //更新Buffer
	virtual void randomize() = 0;
};