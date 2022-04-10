#include <cstdio>
#include <iostream>
#include <fstream>
#include "neuralnetwork.h"
#include "convolutionlayer.h"
#include "fullconnectionlayer.h"
#include "subsamplinglayer.h"
#include "featuremap.h"
#include "image.h"

NeuralNetwork* NeuralNetwork::spNeuralNetwork=nullptr;

double relu(double x)
{
	return x * (x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

NeuralNetwork::~NeuralNetwork()
{
	/*
	for (int i = 0; i < trainNum; i++)
		delete pTrainImage[i];
	for (int i = 0; i < testNum; i++)
		delete pTestImage[i];
	*/
	for (int i = 0; i < layerCount; i++)
		delete mLayers[i];
	for (int i = 0; i < pFeatureMap.size(); i++)
		delete pFeatureMap[i];
	for (int i = 0; i < pError.size(); i++)
		delete pError[i];
}

NeuralNetwork* NeuralNetwork::getInstance()
{
	if (spNeuralNetwork == NULL)
		spNeuralNetwork = new NeuralNetwork;
	return spNeuralNetwork;
}

void NeuralNetwork::releaseInstance()
{
	if (spNeuralNetwork != NULL)
	{
		delete spNeuralNetwork;
		spNeuralNetwork = NULL;
	}
}

void NeuralNetwork::initialize(const string* pconfig_name,
	const string* TRAIN_NAME, const string* TRAINLABEL_NAME,
	const string* TEST_NAME , const string* TESTLABEL_NAME )
{
	ifstream config(pconfig_name->c_str());
	config>>trainNum>>testNum>>alpha>>layerCount;
	config>>image_h>>image_w;
	config >> firstH >> firstW;
	
	Image::setwh(image_h,image_w);

	createFeatureMap(firstH, firstW);
	createError(firstH, firstW);
	
	for (int i = 0; i < layerCount; i++)
	{
		int type; config >> type;
		//cout << type<<"!!!" << endl;
		Layer* curLayer;
		if (type == 1)curLayer = new ConvolutionLayer(config);
		if (type == 2)curLayer = new SubSamplingLayer(config);
		if (type == 3)curLayer = new FullConnectionLayer(config);
		mLayers.push_back(curLayer);
	}

	if (readData(pTrainImage, trainLabel, trainNum, TRAIN_NAME, TRAINLABEL_NAME))
		printf("can't find training data!\n"), system("pause");
	if(readData(pTestImage, testLabel, testNum, TEST_NAME, TESTLABEL_NAME))
		printf("can't find testing data!\n"), system("pause");
	
	/*
	cout << pTrainImage.size() << endl;
	for (int i = 0; i < 28; i++)
	{ 
		for (int j = 0; j < 28;j++)
			if((int)pTrainImage[9999]->data[i][j])cout << 1;
			else cout<<"0";
		cout << endl;
	}
	*/
}
/*config.txt
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte*/

void NeuralNetwork::trainBatch(int batchsize)
{
	// ÅÐ¶ÏbatchsizeÕû³ýtrainNumÂð
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->randomize();
	for (int i = 0; i < trainNum / batchsize; i++)
	{
		cout << i << endl;
		for (int j = 0; j < batchsize; j++)
			train(pTrainImage[i*batchsize+j],&trainLabel[i*batchsize+j]);
		for (int j = 0; j < layerCount; j++)
			mLayers[j]->update(alpha / (double)batchsize);
	}
}

void NeuralNetwork::testBatch()
{
	int percent = 0, correct=0;
	for (int i = 0; i < testNum; i++)
	{
		if (test(pTestImage[i]) ==testLabel[i])correct++;
		if ((double)(i + 1) / testNum * 10 > percent)
		{
			percent = (double)(i + 1) / testNum * 10;
			printf("%d0 %%, accuracy: %llf\n", percent,(double)correct/(i+1));
		}
	}
}

FeatureMap* NeuralNetwork::getFeatureMap(){return pFeatureMap[curFeatureMap++]; }
FeatureMap* NeuralNetwork::getError(){return pError[curError++];}

FeatureMap* NeuralNetwork::createFeatureMap(int height, int width) 
{ 
	++featureMapCount; 
	pFeatureMap.push_back(new FeatureMap(height, width)); 
	return pFeatureMap[featureMapCount-1];
}
FeatureMap* NeuralNetwork::createError(int height, int width) 
{ 
	++errorCount;
	pError.push_back(new FeatureMap(height, width));
	return pError[errorCount-1];
}

void NeuralNetwork::train(Image* image,uint8* label)
{
	for (int i = 0; i < featureMapCount; i++)
		pFeatureMap[i]->clear();
	for (int i = 0; i < errorCount; i++)
		pError[i]->clear();
	image->transform(pFeatureMap[0],firstH,firstW);
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->forward(relu);
	softmax(*label);
	for (int i = layerCount - 1; i >= 0; i--)
		mLayers[i]->backward(relugrad);
}
uint8 NeuralNetwork::getResult()
{
	int result = curFeatureMap;
	for (int i = curFeatureMap; i < featureMapCount; i++)
	{
		cout << pFeatureMap[i]->data[0][0] << ' ';
		if (pFeatureMap[result]->data[0][0] < pFeatureMap[i]->data[0][0])
			result = i;
	}
	cout << result - curFeatureMap << endl;
	return result - curFeatureMap;
}

uint8 NeuralNetwork::test(Image* image)
{
	for (int i = 0; i < featureMapCount; i++)
		pFeatureMap[i]->clear();
	for (int i = 0; i < errorCount; i++)
		pError[i]->clear();
	image->transform(pFeatureMap[0],firstH,firstW);
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->forward(relu);
	return getResult();
}

bool NeuralNetwork::readData(vector<Image*> &image, vector<uint8> &label, int train_or_test_count, const string* cData, const string* cLabel)
{
	FILE* fp_image = fopen(cData->c_str(), "rb");
	FILE* fp_label = fopen(cLabel->c_str(), "rb");
	if (!fp_image || !fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);

	uint8* tImage = new uint8[train_or_test_count*Image::sh*Image::sw];
	fread(tImage, Image::sh * Image::sw * train_or_test_count, 1, fp_image);
	for (int i = 0; i < train_or_test_count; i++)
	{ 
		image.push_back(new Image());

		for (int j = 0; j < Image::sh; j++)
			for (int k = 0; k < Image::sw; k++)
				image[i]->data[j][k] = tImage[i* Image::sh * Image::sw+j*Image::sw+k];
	}

	uint8* tplabel = new uint8[train_or_test_count];
	fread(tplabel, train_or_test_count, 1, fp_label);
	for (int i = 0; i < train_or_test_count;i++)
		label.push_back(tplabel[i]);

	fclose(fp_image);
	fclose(fp_label);
	return 0;
}
void NeuralNetwork::softmax(uint8 label)
{
	double max =pFeatureMap[curFeatureMap+getResult()]->data[0][0];
	double k = 0, inner = 0;

	for (int i = curFeatureMap; i < featureMapCount; ++i)
	{
		pError[curError+i-curFeatureMap]->data[0][0] = exp(pFeatureMap[i]->data[0][0] - max);
		k += pError[curError + i - curFeatureMap]->data[0][0];
	}
	k = 1. / k;
	for (int i = curError; i < errorCount; ++i)
	{
		pError[i]->data[0][0] *= k;
		inner -= (pError[i]->data[0][0])* (pError[i]->data[0][0]);
	}
	inner += pError[label+curError]->data[0][0];
	for (int i = curError; i < errorCount; ++i)
	{
		pError[i]->data[0][0] *= (i == label+curError) - pError[i]->data[0][0] - inner;
	}
}
uint8 NeuralNetwork::testSingle(Image* image)
{
	return test(image);
}
