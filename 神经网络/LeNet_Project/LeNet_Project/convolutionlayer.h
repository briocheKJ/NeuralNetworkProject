#pragma once

#include "layer.h"

class FeatureMap;
class Kernel;

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(int iN, int oN, int _h, int _w); //构造时开kernels,deltas,connection空间
	~ConvolutionLayer();
	void init();
	void forward();
	void backward();

private:
	FeatureMap* inputs; //指向输入特征图数组
	FeatureMap* outputs; //指向输出特征图数组
	Kernel* kernels; //指向卷积核数组
	FeatureMap* inErrors; //向前传递的误差矩阵
	FeatureMap* outErrors; //从后传来的误差矩阵
	Kernel* deltas; //？？？

	int inputN; //输入个数
	int outputN; //输出个数
	int height; //卷积核高
	int width; //卷积核宽
	bool** connection; //输入和输出连接方式
};