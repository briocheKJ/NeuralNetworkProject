#include "neuralnetwork.h"
#include "convolutionlayer.h"
#include "featuremap.h"
#include "kernel.h"
#include <iostream>
#include <cstring>

void convolute_valid(FeatureMap* src, Kernel* conv, int index, FeatureMap* des, int step)
{
	int dh = des->h;
	int dw = des->w;
	int ch = conv->height;
	int cw = conv->width;
	for (int d0 = 0; d0 < dh; ++d0)
		for (int d1 = 0; d1 < dw; ++d1)
		{
			for (int c0 = 0; c0 < ch; ++c0)
				for (int c1 = 0; c1 < cw; ++c1)
				{
					des->data[d0][d1] += src->data[d0 * step + c0][d1 * step + c1] * conv->w[index][c0][c1];
				}
		}
}

void convolute_full(FeatureMap* src, Kernel* conv, int index, FeatureMap* des, int step)
{
	int sh = src->h;
	int sw = src->w;
	int ch = conv->height;
	int cw = conv->width;
	for (int s0 = 0; s0 < sh; ++s0)
		for (int s1 = 0; s1 < sw; ++s1)
		{
			for (int c0 = 0; c0 < ch; ++c0)
				for (int c1 = 0; c1 < cw; ++c1)
				{
					des->data[s0*step + c0][s1*step + c1] += src->data[s0][s1] * conv->w[index][c0][c1];
				}
		}
}

void convolute_deltas(FeatureMap* src, FeatureMap* des, Kernel* conv, int index, int step)
{
	int dh = des->h;
	int dw = des->w;
	int ch = conv->height;
	int cw = conv->width;
	for (int c0 = 0; c0 < ch; ++c0)
		for (int c1 = 0; c1 < cw; ++c1)
		{
			for (int d0 = 0; d0 < dh; ++d0)
				for (int d1 = 0; d1 < dw; ++d1)
				{
					conv->w[index][c0][c1] += src->data[d0 * step + c0][d1 * step + c1] * des->data[d0][d1];
				}
		}
}

ConvolutionLayer::ConvolutionLayer(ifstream&config)
{
	config >> inh;
	inw = inh; //输入特征图大小
	config>> inputN >> outputN; //输入、输出个数
	connection = new bool* [inputN];
	for (int i = 0; i < inputN; i++)
		connection[i] = new bool[outputN];
	init(config);
}

ConvolutionLayer::~ConvolutionLayer()
{
	for (int i = 0; i < outputN; i++)
	{
		delete kernels.back();
		
		delete buffer.back();
		kernels.pop_back();
		buffer.pop_back();

		for (int j = 0; j < THREAD_NUM; j++)
		{
			delete deltas[j].back();
			deltas[j].pop_back();
		}
	}
	for (int i = 0; i < inputN; i++)
		delete[] connection[i];
	delete[] connection;
}

void ConvolutionLayer::init(ifstream&config)
{
	NeuralNetwork* NN = NeuralNetwork::getInstance();

	std::string nows;
	for (int i = 0; i < inputN; i++)
	{
		config >> nows; 
		if (nows[0] == 'f') break;
		for (int j = 0; j < outputN; j++)
			connection[i][j] = nows[j] - '0';
	}

	if (nows[0] == 'f')
	{
		for (int i = 0; i < inputN; i++)
			for (int j = 0; j < outputN; j++)
				connection[i][j] = true;
	}

	config >> height;
	width = height; //卷积核大小
	config >> step >> e; //滑动步长，边扩展

	outh = (inh - height) / step + 1;
	outw = (inw - width) / step + 1;
	//算输出特征图大小

	for (int i = 0; i < outputN; i++)
	{
		int cnt = 0;
		for (int j = 0; j < inputN; j++)
			if (connection[j][i]) cnt++;
		kernels.push_back(new Kernel(cnt, height, width));
		buffer.push_back(new Kernel(cnt, height, width));
		buffer[i]->clear();
		for (int j = 0; j < THREAD_NUM; j++)
		{
			deltas[j].push_back(new Kernel(cnt, height, width));
			deltas[j][i]->clear();
		}
	}

	

	for (int i = 0; i < inputN; i++)
		for (int j = 0; j < THREAD_NUM; j++)
			inputs[j].push_back(NN->getFeatureMap(j));
	for (int i = 0; i < inputN; i++)
		for (int j = 0; j < THREAD_NUM; j++)
			inErrors[j].push_back(NN->getError(j));
	for (int i = 0; i < outputN; i++)
		for (int j = 0; j < THREAD_NUM; j++)
			outputs[j].push_back(NN->createFeatureMap(j, outh, outw));
	for (int i = 0; i < outputN; i++)
		for (int j = 0; j < THREAD_NUM; j++)
			outErrors[j].push_back(NN->createError(j, outh, outw));

	randomize();
}

void ConvolutionLayer::forward(int pid, double (*active)(double))
{
	for (int i = 0; i < outputN; i++)
	{
		int cnt = 0;
		for(int j = 0; j < inputN; j++)
			if (connection[j][i])
			{
				convolute_valid(inputs[pid][j], kernels[i], cnt, outputs[pid][i], step);
				cnt++;
			}
	}
	for (int i = 0; i < outputN; i++)
		for (int j = 0; j < outh; j++)
			for (int k = 0; k < outw; k++)
				outputs[pid][i]->data[j][k] = active(outputs[pid][i]->data[j][k] + kernels[i]->b);
}

void ConvolutionLayer::backward(int pid, double (*activegrad)(double))
{
	for (int i = 0; i < outputN; i++)
	{
		int cnt = 0;
		for (int j = 0; j < inputN; j++)
			if (connection[j][i])
			{
				convolute_full(outErrors[pid][i], kernels[i], cnt, inErrors[pid][j], step);
				cnt++;
			}
	}
	for (int i = 0; i < inputN; i++)
	{
		for (int j = 0; j < inh; j++)
			for (int k = 0; k < inw; k++)
				inErrors[pid][i]->data[j][k] *= activegrad(inputs[pid][i]->data[j][k]);
	}
	for (int i = 0; i < outputN; i++)
		for (int j = 0; j < outh; j++)
			for (int k = 0; k < outw; k++)
				deltas[pid][i]->b += outErrors[pid][i]->data[j][k];
	for (int i = 0; i < outputN; i++)
	{
		int cnt = 0;
		for (int j = 0; j < inputN; j++)
			if (connection[j][i])
			{
				convolute_deltas(inputs[pid][j], outErrors[pid][i], deltas[pid][i], cnt, step);
				cnt++;
			}
	}
	/*
	for (int i = 0; i < outputN; i++)
	{
		buffer[i]->update(deltas[pid][i], 1.0);
		deltas[pid][i]->clear();
	}
	*/ //更新buffer应该独立
}

void ConvolutionLayer::update(double alpha)
{
	for (int i = 0; i < outputN; i++)
	{
		kernels[i]->update(buffer[i], alpha);
		buffer[i]->clear();
	}
}

void ConvolutionLayer::updateBuffer(int pid)
{
	for (int i = 0; i < outputN; i++)
	{
		buffer[i]->update(deltas[pid][i], 1.0);
		deltas[pid][i]->clear();
	}
}

void ConvolutionLayer::randomize()
{
	for (int i = 0; i < outputN; i++)
		kernels[i]->randomize(inputN, outputN);
}
