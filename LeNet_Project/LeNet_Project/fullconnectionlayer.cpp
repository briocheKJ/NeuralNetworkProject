#include "neuralnetwork.h"
#include "fullconnectionlayer.h"
#include "featuremap.h"
#include <iostream>
using namespace std;

FullConnectionLayer::FullConnectionLayer(ifstream& config)
{
	config >> inh;
	inw = inh; //输入特征图大小
	config >> inputN >> outputN; //输入、输出个数
	outw = outh = 1;

	w = new double* [inputN];
	Wbuffer = new double* [inputN];
	for (int i = 0; i < inputN; i++) {
		w[i] = new double[outputN];
		Wbuffer[i] = new double[outputN]{0};
	}
	b = new double[outputN];
	Bbuffer = new double[outputN]{0};
	
	for (int pid = 0; pid < THREAD_NUM; pid++)
	{
		wDeltas[pid] = new double* [inputN];
		for (int i = 0; i < inputN; i++) {
			wDeltas[pid][i] = new double[outputN] {0};
		}
		bDeltas[pid] = new double[outputN] {0};
		z[pid] = new double[outputN];
	}

	init(config);
}
FullConnectionLayer::~FullConnectionLayer() {
	delete[]b;
	delete[]Bbuffer;
	for (int i = 0; i < inputN; i++) {
		delete[]w[i];
		delete[]Wbuffer[i];
	}
	delete[]w;
	delete[]Wbuffer;

	for (int pid = 0; pid < THREAD_NUM; pid++)
	{
		for (int i = 0; i < inputN; i++) {
			delete[] wDeltas[pid][i];
		}
		delete[] wDeltas[pid];
		delete[] bDeltas[pid];
		delete[] z[pid];
	}
}
void FullConnectionLayer::init(ifstream& config)
{
	NeuralNetwork* NN = NeuralNetwork::getInstance();

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
void FullConnectionLayer::randomize() {
	for (int i = 0; i < outputN; i++) {
		b[i] = 0.0;
	}
	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			w[i][j] = rand() * (2. / RAND_MAX) - 1;
			w[i][j] *= sqrt(6.0 / (double)(inputN + outputN));
		}
	}
}
void FullConnectionLayer::forward(int pid, double (*active)(double)) {  //计算outputs
	for (int i = 0; i < outputN; i++) {
		// 下一层第i个神经元
		double ans = 0;
		for (int j = 0; j < inputN; j++) {
			ans = ans + w[j][i] * inputs[pid][j]->data[0][0];
		}
		ans = ans + b[i];
		z[pid][i] = ans;
		ans = active(z[pid][i]);
		outputs[pid][i]->data[0][0] = ans;
	}
}
void FullConnectionLayer::backward(int pid, double (*activegrad)(double)) {
	for (int i = 0; i < inputN; i++) {
		//第i个(l-1)层神经元
		double sum = 0;
		for (int j = 0; j < outputN; j++) {
			sum = sum + outErrors[pid][j]->data[0][0] * w[i][j]*activegrad(inputs[pid][i]->data[0][0]);
		}
		inErrors[pid][i]->data[0][0] = sum;
	}

	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			wDeltas[pid][i][j] += outErrors[pid][j]->data[0][0] * activegrad(inputs[pid][i]->data[0][0]) * inputs[pid][i]->data[0][0];
		}
	}
	for (int i = 0; i < outputN; i++) {
		bDeltas[pid][i] += outErrors[pid][i]->data[0][0];
	}

	/*
	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			Wbuffer[i][j] += outErrors[pid][j]->data[0][0] * activegrad(inputs[i]->data[0][0]) * inputs[i]->data[0][0];
		}
	}
	for (int i = 0; i < outputN; i++) {
		Bbuffer[i] += outErrors[pid][i]->data[0][0];
	}
	*/
}

void FullConnectionLayer::updateBuffer(int pid)
{
	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			Wbuffer[i][j] += wDeltas[pid][i][j];
			wDeltas[pid][i][j] = 0.0;
		}
	}
	for (int i = 0; i < outputN; i++) {
		Bbuffer[i] += bDeltas[pid][i];
		bDeltas[pid][i] = 0.0;
	}
}

void FullConnectionLayer::update(double alpha) {
	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			w[i][j] = w[i][j] + alpha * Wbuffer[i][j];
			Wbuffer[i][j] = 0;
		}
	}
	for (int i = 0; i < outputN; i++) {
		b[i] = b[i] + Bbuffer[i] * alpha;
		Bbuffer[i] = 0;
	}
}
