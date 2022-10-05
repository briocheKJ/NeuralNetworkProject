#pragma once

#include "layer.h"
#include <vector>
#include <fstream>

class FeatureMap;
class Kernel;

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(ifstream&); //����ʱ��kernels,deltas,connection�ռ�
	~ConvolutionLayer();
	void init(ifstream&);
	void forward(int pid, double (*active)(double)); //����outputs
	void backward(int pid, double (*activegrad)(double)); //����errors,deltas
	void update(double alpha); //��buffer����kernels��ѧϰ��alpha
	void updateBuffer(int pid);
	void randomize(); //�����kernels����

private:
	std::vector<FeatureMap*> inputs[THREAD_NUM]; //��������ͼ����
	std::vector<FeatureMap*> outputs[THREAD_NUM]; //�������ͼ����
	std::vector<Kernel*> kernels; //���������
	std::vector<FeatureMap*> inErrors[THREAD_NUM]; //��ǰ���ݵ�������
	std::vector<FeatureMap*> outErrors[THREAD_NUM]; //�Ӻ�����������
	std::vector<Kernel*> deltas[THREAD_NUM]; //����������
	std::vector<Kernel*> buffer; //buffer

	int inputN; //�������
	int outputN; //�������
	int height; //����˸�
	int width; //����˿�
	int step; //��������
	int e; //����չ��
	int inh; //�����
	int inw; //�����
	int outh; //�����
	int outw; //�����
	bool** connection; //�����������ӷ�ʽ
};