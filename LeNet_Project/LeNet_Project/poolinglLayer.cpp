#include "neuralnetwork.h"
#include "poolingLayer.h"
#include "featuremap.h"
#include <iostream>
using namespace std;
PoolingLayer::PoolingLayer()
{
	cin >> inh;
	inw = inh; //输入特征图大小
	cin >> inputN >> outputN; //输入、输出个数
	group = outputN/inputN;
	cin >> f;
	outh = outw = inh / f;
	init();
}
PoolingLayer::~PoolingLayer() {
	//for (int i = 0; i <inputN; i++) delete inputs[i];
	//for (int i = 0; i < outputN; i++) delete outputs[i];
}
void PoolingLayer::init() {
	NeuralNetwork* NN = NeuralNetwork::getInstance();
    
	for (int i = 0; i < inputN; i++)
		inputs.push_back(NN->getFeatureMap());
	for (int i = 0; i < inputN; i++)
		inErrors.push_back(NN->getError());
	for (int i = 0; i < outputN; i++)
		outputs.push_back(NN->createFeatureMap(outh, outh));
	for (int i = 0; i < outputN; i++)
		outErrors.push_back(NN->createError(outh, outw));
    //randomize();
}
int PoolingLayer::max(int index,int i, int k, int j) {
	// i: 第i个输入   k：第k行   j: 第j列
	int max = 0; int maxcol, maxrow;
	for (int row = k * f; row < (k + 1) * f; k++) {
		for (int col = j * f; col < (j + 1) * f; col++) {
			if (inputs[i]->data[col][row] > max) {
				max = inputs[i]->data[col][row];
				maxcol = col; maxrow = row;
			}
		}
	}
	for (int row = k * f; row < (k + 1) * f; k++) {
		for (int col = j * f; col < (j + 1) * f; col++) {
			inErrors[index]->data[row][col] = 0;
		}
	}
	inErrors[index]->data[maxrow][maxcol] = max;
	return max;
}
void PoolingLayer::setz(int i, int index) {
	for (int k =0; k <outh; k++) {
		for (int j = 0; j < outh; j++) {
			outputs[index]->data[k][j] = max(index,i, k, j);
		}
	}
}
 void PoolingLayer::forward(double (*active)(double)) {
	for (int i = 0; i <inputN; i++) {
		for (int j = 0; j < group;j++) {
			int index; // 第index个outputs
			index = (i)*group + j;
			setz(i,index);
		}
	}
 }
 
virtual void backward(double (*activegrad)(double));