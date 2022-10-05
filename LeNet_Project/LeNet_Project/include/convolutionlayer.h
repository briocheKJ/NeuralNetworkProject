#pragma once

#include "layer.h"
#include <vector>
#include <fstream>

class FeatureMap;
class Kernel;

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(ifstream&); //构造时开kernels,deltas,connection空间
	~ConvolutionLayer();
	void init(ifstream&);
	void forward(int pid, double (*active)(double)); //计算outputs
	void backward(int pid, double (*activegrad)(double)); //计算errors,deltas
	void update(double alpha); //用buffer更新kernels，学习率alpha
	void updateBuffer(int pid);
	void randomize(); //随机化kernels参数

private:
	std::vector<FeatureMap*> inputs[THREAD_NUM]; //输入特征图数组
	std::vector<FeatureMap*> outputs[THREAD_NUM]; //输出特征图数组
	std::vector<Kernel*> kernels; //卷积核数组
	std::vector<FeatureMap*> inErrors[THREAD_NUM]; //向前传递的误差矩阵
	std::vector<FeatureMap*> outErrors[THREAD_NUM]; //从后传来的误差矩阵
	std::vector<Kernel*> deltas[THREAD_NUM]; //参数更新量
	std::vector<Kernel*> buffer; //buffer

	int inputN; //输入个数
	int outputN; //输出个数
	int height; //卷积核高
	int width; //卷积核宽
	int step; //滑动步长
	int e; //边扩展？
	int inh; //输入高
	int inw; //输入宽
	int outh; //输出高
	int outw; //输出宽
	bool** connection; //输入和输出连接方式
};