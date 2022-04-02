#pragma once

#include "layer.h"
#include <vector>

class FeatureMap;
class Kernel;

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(); //����ʱ��kernels,deltas,connection�ռ�
	~ConvolutionLayer();
	void init();
	void forward(double (*active)(double)); //����outputs
	void backward(double (*activegrad)(double)); //����errors,deltas
	void update(double alpha); //��buffer����kernels��ѧϰ��alpha
	void randomize(); //�����kernels����

private:
	std::vector<FeatureMap*> inputs; //ָ����������ͼ����
	std::vector<FeatureMap*> outputs; //ָ���������ͼ����
	std::vector<Kernel*> kernels; //ָ����������
	std::vector<FeatureMap*> inErrors; //��ǰ���ݵ�������
	std::vector<FeatureMap*> outErrors; //�Ӻ�����������
	std::vector<Kernel*> deltas; //����������
	std::vector<Kernel*> buffer; //experience replay buffer

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