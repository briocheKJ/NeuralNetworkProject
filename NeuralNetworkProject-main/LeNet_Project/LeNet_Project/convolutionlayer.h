#pragma once

#include "layer.h"

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
	FeatureMap* inputs; //ָ����������ͼ����
	FeatureMap* outputs; //ָ���������ͼ����
	Kernel* kernels; //ָ����������
	FeatureMap* inErrors; //��ǰ���ݵ�������
	FeatureMap* outErrors; //�Ӻ�����������
	Kernel* deltas; //������

	int inputN; //�������
	int outputN; //�������
	int height; //����˸�
	int width; //����˿�
	bool** connection; //�����������ӷ�ʽ
};