#pragma once
#include "layer.h"
#include <vector>
class FullConnectionLayer : public Layer{
public:
	FullConnectionLayer(ifstream&config); //����ʱ��kernels,deltas,connection�ռ�
	~FullConnectionLayer();
	void init(ifstream&);
	void forward(double (*active)(double)); //����outputs
	void backward(double (*activegrad)(double)); //����errors,deltas
	void update(double alpha); //��buffer����kernels��ѧϰ��alpha
	void randomize() {
		for (int i = 0; i < inputN; i++) b[i] = 0;
		for (int i = 0; i < inputN; i++) {
			for (int j = 0; j < outputN; j++) {
				w[i][j] = rand() * (2. / RAND_MAX) - 1;
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
	double* b;



};