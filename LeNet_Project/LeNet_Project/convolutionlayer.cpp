#include "neuralnetwork.h"
#include "convolutionlayer.h"
#include "featuremap.h"
#include "kernel.h"
#include <iostream>
#include <cstring>

void convolute_valid(FeatureMap* src, Kernel* conv, int index, FeatureMap* des)
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
					des->data[d0][d1] += src->data[d0 + c0][d1 + c1] * conv->w[index][c0][c1];
				}
		}
}

void convolute_full(FeatureMap* src, Kernel* conv, int index, FeatureMap* des)
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
					des->data[s0 + c0][s1 + c1] += src->data[s0][s1] * conv->w[index][c0][c1];
				}
		}
}

ConvolutionLayer::ConvolutionLayer(int iN, int oN, int _h, int _w) :inputN(iN), outputN(oN), height(_h), width(_w)
{
	for (int i = 0; i < outputN; i++)
		kernels.push_back(new Kernel(inputN, height, width));
	for (int i = 0; i < outputN; i++)
		deltas.push_back(new Kernel(inputN, height, width));
	connection = new bool* [inputN];
	for (int i = 0; i < inputN; i++)
		connection[i] = new bool[outputN];
	init();
}

ConvolutionLayer::~ConvolutionLayer()
{
	for (int i = 0; i < outputN; i++)
	{
		delete kernels.back();
		kernels.pop_back();
	}
	for (int i = 0; i < outputN; i++)
	{
		delete deltas.back();
		deltas.pop_back();
	}
	for (int i = 0; i < inputN; i++)
		delete[] connection[i];
	delete[] connection;
}

void ConvolutionLayer::init()
{
	NeuralNetwork* NN = NeuralNetwork::getInstance();
	cin >> inh >> outh;
	inw = inh;
	outw = outh;

	std::string nows;
	for (int i = 0; i < inputN; i++)
	{
		cin >> nows;
		for (int j = 0; j < outputN; j++)
			connection[i][j] = nows[j] - '0';
	}
	
	for (int i = 0; i < inputN; i++)
		inputs.push_back(NN->getFeatureMap());
	for (int i = 0; i < inputN; i++)
		inErrors.push_back(NN->getError());
	for (int i = 0; i < outputN; i++)
		outputs.push_back(NN->createFeatureMap(outh,outw));
	for (int i = 0; i < outputN; i++)
		outErrors.push_back(NN->createFeatureMap(outh,outw));
}

void ConvolutionLayer::forward()
{
}

void ConvolutionLayer::backward()
{

}