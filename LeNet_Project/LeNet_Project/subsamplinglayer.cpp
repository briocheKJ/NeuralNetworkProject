#include "neuralnetwork.h"
#include "subsamplinglayer.h"
#include "featuremap.h"
#include <iostream>
using namespace std;
SubSamplingLayer::SubSamplingLayer(ifstream& config)
{
	config >> inh;
	inw = inh; //��������ͼ��С
	config >> inputN >> outputN; //���롢�������
	config >> f;
	outh = inh / f;
	outw = inw / f;
	mark = new int** [inputN];
	for (int i = 0; i < inputN; i++) {
		mark[i] = new int* [inh];
		for (int j = 0; j < inh; j++) {
			mark[i][j] = new int[inw] {0};
		}
	}
	init(config);
}
SubSamplingLayer::~SubSamplingLayer() {
	for (int i = 0; i < inputN; i++) {
		for (int row = 0; row < inh; row++) {
			delete[] mark[i][row];
		}
		delete[] mark[i];
	}
	delete[] mark;
}
void SubSamplingLayer::init(ifstream& config) {
	NeuralNetwork* NN = NeuralNetwork::getInstance();
    for (int i = 0; i < inputN; i++)
		inputs.push_back(NN->getFeatureMap());
	for (int i = 0; i < inputN; i++)
		inErrors.push_back(NN->getError());
	for (int i = 0; i < outputN; i++)
		outputs.push_back(NN->createFeatureMap(outh, outw));
	for (int i = 0; i < outputN; i++)
		outErrors.push_back(NN->createError(outh, outw));
	for (int i = 0; i < inputN; i++) {
		for (int row = 0; row < inh; row++) {
			for (int col = 0; col < inw; col++) {
				inErrors[i]->data[row][col] = 0;
			}
		}
	}
}
void SubSamplingLayer::forward(double (*active)(double)) {
	for (int i = 0; i < inputN; i++) {
		// ��i�����featuremap
		for (int row = 0; row < outh; row++) {
			for (int col = 0; col < outh; col++) {
				outputs[i]->data[row][col] = Max(i,row,col);
			}
		}
	}
}
double  SubSamplingLayer::Max(int i,int row,int col) {
	// ��i��input�� output�ĵ�row�е�col��
	double max = 0; int maxrow=0, maxcol=0;
	for (int Irow = row * f; Irow <= (row + 1) * f - 1; Irow++) {
		for (int Icol = col * f; Icol <= (col + 1) * f - 1; Icol++) {
			if (inputs[i]->data[Irow][Icol] > max) {
				max = inputs[i]->data[Irow][Icol];
				maxrow = Irow; maxcol = Icol;
			}
		}
	}
	mark[i][maxrow][maxcol] = 1;
	return max;
}

void SubSamplingLayer::backward(double (*activegrad)(double)) {
	for (int i = 0; i < inputN; i++) {
	    // ���i��Outerrors
		for (int row = 0; row < outh; row++) {
			for (int col = 0; col < outh; col++) {
				double data = outErrors[i]->data[row][col];
				SetInError(i, data, row, col);
			}
		}
	}
}
void SubSamplingLayer::SetInError(int i, double data, int row, int col) {
	// д��i��InError��
	for (int Irow = row * f; Irow <= (row + 1) * f - 1; Irow++) {
		for (int Icol = col * f; Icol <= (col + 1) * f - 1; Icol++) {
			if (mark[i][Irow][Icol] == 1) {
				inErrors[i]->data[row][col] = data; 
				return; 
			}
		}
	}
}
