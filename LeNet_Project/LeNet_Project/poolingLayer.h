#pragma once
#include"layer.h"
#include<vector>
class PoolingLayer :public Layer{
	
public: 
        PoolingLayer();
        ~PoolingLayer();
        virtual void init();
        virtual void forward(double (*active)(double)); 
		virtual void backward(double (*activegrad)(double)) {
			;
		}
		virtual void update(double alpha) {
			;
		}
        //virtual void randomize(); 

private:
	void setz(int i, int index);//
	int max(int index,int i, int k, int j);
	std::vector<FeatureMap*> inputs; //指向输入特征图数组
	std::vector<FeatureMap*> outputs; //指向输出特征图数组
	std::vector<FeatureMap*> inErrors; //向前传递的误差矩阵
	std::vector<FeatureMap*> outErrors; //从后传来的误差矩阵
	
	int inputN; //输入个数
	int outputN; //输出个数
	int group;//组数
	int f;//缩放比例
	int inh; //输入高
	int inw; //输入宽
	int outh; //输出高
	int outw; //输出宽



};