#pragma once
#include "layer.h"
#include <vector>
class FullyConnectedLayer : public Layer{
public:
	FullyConnectedLayer(); //����ʱ��kernels,deltas,connection�ռ�
	~FullyConnectedLayer();
	void init();
	void forward(double (*active)(double)); //����outputs
	void backward(double (*activegrad)(double)); //����errors,deltas
	void update(double alpha); //��buffer����kernels��ѧϰ��alpha
	void randomize() {
		for (int i = 0; i < outputN; i++) b[i] = 0;
		for (int i = 0; i < outputN; i++) {
			for (int j = 0; j < inputN; j++) {
				w[i][j] = 1;
			}
		}

	}


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
	double** Wbuffet;
	double* Bbuffet;
	double** w;
	double *b;

};