#pragma once
#include"layer.h"
#include<vector>
class PoolingLayer :public Layer{
	
public: 
        PoolingLayer();
        ~PoolingLayer();
        virtual void init();
        virtual void forward(double (*active)(double)); 
		virtual void backward(double (*activegrad)(double)) {
			;
		}
		virtual void update(double alpha) {
			;
		}
        //virtual void randomize(); 

private:
	void setz(int i, int index);//
	int max(int index,int i, int k, int j);
	std::vector<FeatureMap*> inputs; //ָ����������ͼ����
	std::vector<FeatureMap*> outputs; //ָ���������ͼ����
	std::vector<FeatureMap*> inErrors; //��ǰ���ݵ�������
	std::vector<FeatureMap*> outErrors; //�Ӻ�����������
	
	int inputN; //�������
	int outputN; //�������
	int group;//����
	int f;//���ű���
	int inh; //�����
	int inw; //�����
	int outh; //�����
	int outw; //�����



};