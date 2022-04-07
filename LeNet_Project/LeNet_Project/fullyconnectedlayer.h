#pragma once
#include "layer.h"
#include <vector>
class FullyConnectedLayer : public Layer{
public:
	FullyConnectedLayer(); //构造时开kernels,deltas,connection空间
	~FullyConnectedLayer();
	void init();
	void forward(double (*active)(double)); //计算outputs
	void backward(double (*activegrad)(double)); //计算errors,deltas
	void update(double alpha); //用buffer更新kernels，学习率alpha
	void randomize() {
		for (int i = 0; i < outputN; i++) b[i] = 0;
		for (int i = 0; i < outputN; i++) {
			for (int j = 0; j < inputN; j++) {
				w[i][j] = 1;
			}
		}

	}


private:
	int inputN; //输入个数
	int outputN; //输出个数
	int inh; //输入高
	int inw; //输入宽
	int outh; //输出高
	int outw; //输出宽


private:
	std::vector<FeatureMap*> inputs; //指向输入特征图数组
	std::vector<FeatureMap*> outputs; //指向输出特征图数组
	std::vector<FeatureMap*> inErrors; //向前传递的误差矩阵
	std::vector<FeatureMap*> outErrors; //从后传来的误差矩阵
	double** Wbuffet;
	double* Bbuffet;
	double** w;
	double *b;

};
