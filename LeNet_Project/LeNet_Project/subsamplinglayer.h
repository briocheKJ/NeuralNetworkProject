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
	std::vector<FeatureMap*> inputs; //ָ����������ͼ����
	std::vector<FeatureMap*> outputs; //ָ���������ͼ����
	std::vector<FeatureMap*> inErrors; //��ǰ���ݵ�������
	std::vector<FeatureMap*> outErrors; //�Ӻ�����������
    
    int inputN; //�������
	int outputN; //�������
	int f;//���ű���
	int inh; //�����
	int inw; //�����
	int outh; //�����
	int outw; //�����
	int*** mark;
};



