#pragma once
#include <fstream>
class FeatureMap;

const int THREAD_NUM=16; //�߳���

class Layer
{
public:
	virtual ~Layer() {}
	virtual void init(std::ifstream&) = 0;
	virtual void forward(int pid, double (*active)(double)) = 0; //pid�����̱߳�š�
	virtual void backward(int pid, double (*activegrad)(double)) = 0; //pid�����̱߳�š�
	virtual void update(double alpha) = 0; //��Buffer���¿�ѵ��������ѧϰ��alpha
	virtual void updateBuffer(int pid) = 0; //����Buffer
	virtual void randomize() = 0;
};