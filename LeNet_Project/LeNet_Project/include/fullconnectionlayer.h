#pragma once
#include "layer.h"
#include <vector>
#include <iostream>
class FullConnectionLayer : public Layer{
public:
	FullConnectionLayer(ifstream& config); //构造时开kernels,deltas,connection空间
	~FullConnectionLayer();
	void init(ifstream& config);
	void forward(int pid, double (*active)(double)); //计算outputs
	void backward(int pid, double (*activegrad)(double)); //计算errors,deltas
	void update(double alpha); //用buffer更新kernels，学习率alpha
	void updateBuffer(int pid);
	void randomize();

	


private:
	int inputN; //输入个数
	int outputN; //输出个数
	int inh; //输入高
	int inw; //输入宽
	int outh; //输出高
	int outw; //输出宽


private:
	std::vector<FeatureMap*> inputs[THREAD_NUM]; //指向输入特征图数组
	std::vector<FeatureMap*> outputs[THREAD_NUM]; //指向输出特征图数组
	std::vector<FeatureMap*> inErrors[THREAD_NUM]; //向前传递的误差矩阵
	std::vector<FeatureMap*> outErrors[THREAD_NUM]; //从后传来的误差矩阵
	double** Wbuffer;
	double* Bbuffer;
	double** w;
	double* b;

	double** wDeltas[THREAD_NUM];
	double* bDeltas[THREAD_NUM];

	double* z[THREAD_NUM];
};