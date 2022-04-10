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
		Wbuffer[i] = new double[outputN];
	}
	b = new double[outputN];
	Bbuffer = new double[outputN];
	z = new double[outputN];
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
	delete[]z;
}
void FullConnectionLayer::init(ifstream& config)
{
	NeuralNetwork* NN = NeuralNetwork::getInstance();
	for (int i = 0; i < inputN; i++)
		inputs.push_back(NN->getFeatureMap());
	for (int i = 0; i < inputN; i++)
		inErrors.push_back(NN->getError());
	for (int i = 0; i < outputN; i++)
		outputs.push_back(NN->createFeatureMap(outh, outw));
	for (int i = 0; i < outputN; i++)
		outErrors.push_back(NN->createError(outh, outw));
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
void FullConnectionLayer::forward(double (*active)(double)) {  //计算outputs
	for (int i = 0; i < outputN; i++) {
		// 下一层第i个神经元
		double ans = 0;
		for (int j = 0; j < inputN; j++) {
			ans = ans + w[j][i] * inputs[j]->data[0][0];
		}
		ans = ans + b[i];
		z[i] = ans;
		ans = active(z[i]);
		outputs[i]->data[0][0] = ans;
	}
}
void FullConnectionLayer::backward(double (*activegrad)(double)) {
	for (int i = 0; i < inputN; i++) {
		//第i个(l-1)层神经元
		double sum = 0;
		for (int j = 0; j < outputN; j++) {
			sum = sum + outErrors[j]->data[0][0] * w[i][j]*activegrad(z[j]);
		}
		inErrors[i]->data[0][0] = sum;
	}
	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			Wbuffer[i][j] = outErrors[j]->data[0][0] * activegrad(z[j])*inputs[i]->data[0][0];
		}
	}
	for (int i = 0; i < outputN; i++) {
		Bbuffer[i] = outErrors[i]->data[0][0]* activegrad(z[i]);
	}
}
void FullConnectionLayer::update(double alpha) {
	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			w[i][j] = w[i][j] - alpha * Wbuffer[i][j];
		}
	}
	for (int i = 0; i < outputN; i++) {
		b[i] = b[i] - Bbuffer[i] * alpha;
	}
}