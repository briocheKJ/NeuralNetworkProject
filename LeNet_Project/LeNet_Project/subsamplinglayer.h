#pragma once
#include "layer.h"
#include <vector>
class SubSamplingLayer : public Layer
{
public:
	SubSamplingLayer(ifstream&);
	~SubSamplingLayer();
	virtual void init(ifstream&);
	virtual void forward(double (*active)(double));
	virtual void backward(double (*activegrad)(double));
	virtual void update(double alpha) {}
	void SetInError(int i, double data, int row, int col);
	virtual void randomize() {}

private:
	double Max(int i, int row, int col);
	std::vector<FeatureMap*> inputs; //指向输入特征图数组
	std::vector<FeatureMap*> outputs; //指向输出特征图数组
	std::vector<FeatureMap*> inErrors; //向前传递的误差矩阵
	std::vector<FeatureMap*> outErrors; //从后传来的误差矩阵
    
    int inputN; //输入个数
	int outputN; //输出个数
	int f;//缩放比例
	int inh; //输入高
	int inw; //输入宽
	int outh; //输出高
	int outw; //输出宽
	int*** mark;
};



