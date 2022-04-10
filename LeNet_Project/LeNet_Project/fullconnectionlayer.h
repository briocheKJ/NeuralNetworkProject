#pragma once
#include "layer.h"
#include <vector>
#include <iostream>
class FullConnectionLayer : public Layer{
public:
	FullConnectionLayer(ifstream& config); //����ʱ��kernels,deltas,connection�ռ�
	~FullConnectionLayer();
	void init(ifstream& config);
	void forward(double (*active)(double)); //����outputs
	void backward(double (*activegrad)(double)); //����errors,deltas
	void update(double alpha); //��buffer����kernels��ѧϰ��alpha
	void randomize();

	


private:
	int inputN; //�������
	int outputN; //�������
	int inh; //�����
	int inw; //�����
	int outh; //�����
	int outw; //�����


private:
	std::vector<FeatureMap*> inputs; //ָ����������ͼ����
	std::vector<FeatureMap*> outputs; //ָ���������ͼ����
	std::vector<FeatureMap*> inErrors; //��ǰ���ݵ�������
	std::vector<FeatureMap*> outErrors; //�Ӻ�����������
	double** Wbuffer;
	double* Bbuffer;
	double** w;
	double* b;
	double* z;
};