#pragma once
#include "layer.h"
#include <vector>
#include <iostream>
class FullConnectionLayer : public Layer{
public:
	FullConnectionLayer(ifstream& config); //����ʱ��kernels,deltas,connection�ռ�
	~FullConnectionLayer();
	void init(ifstream& config);
	void forward(int pid, double (*active)(double)); //����outputs
	void backward(int pid, double (*activegrad)(double)); //����errors,deltas
	void update(double alpha); //��buffer����kernels��ѧϰ��alpha
	void updateBuffer(int pid);
	void randomize();

	


private:
	int inputN; //�������
	int outputN; //�������
	int inh; //�����
	int inw; //�����
	int outh; //�����
	int outw; //�����


private:
	std::vector<FeatureMap*> inputs[THREAD_NUM]; //ָ����������ͼ����
	std::vector<FeatureMap*> outputs[THREAD_NUM]; //ָ���������ͼ����
	std::vector<FeatureMap*> inErrors[THREAD_NUM]; //��ǰ���ݵ�������
	std::vector<FeatureMap*> outErrors[THREAD_NUM]; //�Ӻ�����������
	double** Wbuffer;
	double* Bbuffer;
	double** w;
	double* b;

	double** wDeltas[THREAD_NUM];
	double* bDeltas[THREAD_NUM];

	double* z[THREAD_NUM];
};