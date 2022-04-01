#pragma once

#include "layer.h"
#include <vector>

class FeatureMap;
class Kernel;

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(int iN, int oN, int _h, int _w); //����ʱ��kernels,deltas,connection�ռ�
	~ConvolutionLayer();
	void init();
	void forward();
	void backward();

private:
	std::vector<FeatureMap*> inputs; //ָ����������ͼ����
	std::vector<FeatureMap*> outputs; //ָ���������ͼ����
	std::vector<Kernel*> kernels; //ָ����������
	std::vector<FeatureMap*> inErrors; //��ǰ���ݵ�������
	std::vector<FeatureMap*> outErrors; //�Ӻ�����������
	std::vector<Kernel*> deltas; //������

	int inputN; //�������
	int outputN; //�������
	int height; //����˸�
	int width; //����˿�
	int inh; //�����
	int inw; //�����
	int outh; //�����
	int outw; //�����
	bool** connection; //�����������ӷ�ʽ
};