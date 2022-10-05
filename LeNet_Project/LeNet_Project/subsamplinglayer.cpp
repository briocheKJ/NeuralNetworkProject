#include "neuralnetwork.h"
#include "subsamplinglayer.h"
#include "featuremap.h"
#include <iostream>
using namespace std;
SubSamplingLayer::SubSamplingLayer(ifstream& config)
{
	config >> inh;
	inw = inh; //输入特征图大小
	config >> inputN >> outputN; //输入、输出个数
	config >> f;
	outh = inh / f;
	outw = inw / f;
	for (int pid = 0; pid < THREAD_NUM; pid++)
	{
		mark[pid] = new int** [inputN];
		for (int i = 0; i < inputN; i++) {
			mark[pid][i] = new int* [inh];
			for (int j = 0; j < inh; j++) {
				mark[pid][i][j] = new int[inw] {0};
			}
		}
	}
	init(config);
}
SubSamplingLayer::~SubSamplingLayer() {
	for (int pid = 0; pid < THREAD_NUM; pid++)
	{
		for (int i = 0; i < inputN; i++) {
			for (int row = 0; row < inh; row++) {
				delete[] mark[pid][i][row];
			}
			delete[] mark[pid][i];
		}
		delete[] mark[pid];
	}
}
void SubSamplingLayer::init(ifstream& config) {
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
	for (int i = 0; i < inputN; i++)
		for (int j = 0; j < THREAD_NUM; j++)
			inErrors[j][i]->clear();
}

void SubSamplingLayer::forward(int pid, double (*active)(double)) {
	for (int i = 0; i < inputN; i++) {
		// 第i个输出featuremap
		for (int row = 0; row < outh; row++) {
			for (int col = 0; col < outh; col++) {
				outputs[pid][i]->data[row][col] = Max(pid,i,row,col);
			}
		}
	}
}
double  SubSamplingLayer::Max(int pid,int i,int row,int col) {
	// 第i个input， output的第row行第col列
	double max = 0; int maxrow=0, maxcol=0;
	for (int Irow = row * f; Irow <= (row + 1) * f - 1; Irow++) {
		for (int Icol = col * f; Icol <= (col + 1) * f - 1; Icol++) {
			if (inputs[pid][i]->data[Irow][Icol] > max) {
				max = inputs[pid][i]->data[Irow][Icol];
				maxrow = Irow; maxcol = Icol;
			}
		}
	}
	mark[pid][i][maxrow][maxcol] = 1;
	return max;
}

void SubSamplingLayer::backward(int pid, double (*activegrad)(double)) {
	for (int i = 0; i < inputN; i++) {
	    // 求第i个Outerrors
		for (int row = 0; row < outh; row++) {
			for (int col = 0; col < outh; col++) {
				double data = outErrors[pid][i]->data[row][col];
				SetInError(pid, i, data, row, col);
			}
		}
	}
}
void SubSamplingLayer::SetInError(int pid, int i, double data, int row, int col) {
	// 写第i个InError，
	for (int Irow = row * f; Irow <= (row + 1) * f - 1; Irow++) {
		for (int Icol = col * f; Icol <= (col + 1) * f - 1; Icol++) {
			if (mark[pid][i][Irow][Icol] == 1) {
				inErrors[pid][i]->data[Irow][Icol] = data;
				mark[pid][i][Irow][Icol] = 0;
				return;
			}
			else inErrors[pid][i]->data[Irow][Icol] = 0.0;
		}
	}
}
