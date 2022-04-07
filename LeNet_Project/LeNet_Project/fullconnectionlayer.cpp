#include "neuralnetwork.h"
#include "fullconnectionlayer.h"
#include "featuremap.h"
#include <iostream>
using namespace std;

FullConnectionLayer::FullConnectionLayer()
{
	cin >> inh;
	inw = inh = 1; //输入特征图大小
	cin >> inputN >> outputN; //输入、输出个数
	w = new double* [inputN];
	Wbuffet = new double* [inputN];
	for (int i = 0; i < outputN; i++) {
		w[i] = new double[outputN];
		Wbuffet[i] = new double[outputN];
	}
	b = new double[outputN];
	Bbuffet = new double[outputN];
	init();
}
FullConnectionLayer::~FullConnectionLayer() {
	delete[]b;
	for (int i = 0; i < inputN; i++) {
		delete[]w[i];
	}
	delete[]w;
}
void FullConnectionLayer::init()
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
void FullConnectionLayer::forward(double (*active)(double)) {  //计算outputs
	for (int i = 0; i < inputN; i++) {// 第i个神经元
		double ans = 0;
		for (int j = 0; j < outputN; j++) {
			ans = ans + w[i][j] * inputs[i]->data[0][0];
		}
		ans = ans + b[i];
		ans = active(ans);
		outputs[i]->data[0][0] = ans;
	}
}
void FullConnectionLayer::backward(double (*activegrad)(double)) {
	for (int i = 0; i < inputN; i++) {
		double sum = 0;
		for (int j = 0; j < outputN; j++) {
			sum = sum + outErrors[j]->data[0][0] * w[i][j];
		}
		sum = sum * activegrad(inputs[i]->data[0][0]);
		inErrors[i]->data[0][0] = sum;
	}
	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			Wbuffet[i][j] = outErrors[j]->data[0][0] * outputs[i]->data[0][0];
		}
	}
	for (int i = 0; i < outputN; i++) {
		Bbuffet[i] = outErrors[i]->data[0][0];
	}
}
void FullConnectionLayer::update(double alpha) {
	for (int i = 0; i < inputN; i++) {
		for (int j = 0; j < outputN; j++) {
			w[i][j] = w[i][j] - alpha * Wbuffet[i][j];
		}
	}
	for (int i = 0; i < outputN; i++) {
		b[i] = b[i] - Bbuffet[i] * alpha;
	}
}