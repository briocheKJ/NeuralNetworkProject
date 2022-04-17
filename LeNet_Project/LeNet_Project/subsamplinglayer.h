#pragma once
#include "layer.h"
#include <vector>
class SubSamplingLayer : public Layer
{
public:
	SubSamplingLayer(ifstream&);
	~SubSamplingLayer();
	virtual void init(ifstream&);
	virtual void forward(int pid, double (*active)(double));
	virtual void backward(int pid, double (*activegrad)(double));
	virtual void update(double alpha) {}
	virtual void updateBuffer(int pid) {}
	void SetInError(int pid, int i, double data, int row, int col);
	virtual void randomize() {}

private:
	double Max(int pid, int i, int row, int col);
	std::vector<FeatureMap*> inputs[THREAD_NUM]; //ָ����������ͼ����
	std::vector<FeatureMap*> outputs[THREAD_NUM]; //ָ���������ͼ����
	std::vector<FeatureMap*> inErrors[THREAD_NUM]; //��ǰ���ݵ�������
	std::vector<FeatureMap*> outErrors[THREAD_NUM]; //�Ӻ�����������
    
    int inputN; //�������
	int outputN; //�������
	int f;//���ű���
	int inh; //�����
	int inw; //�����
	int outh; //�����
	int outw; //�����
	int*** mark[THREAD_NUM];
};



