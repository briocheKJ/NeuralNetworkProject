#include <cstdio>

#include "neuralnetwork.h"
#include "convolutionlayer.h"
#include "fullconnectionlayer.h"
#include "subsamplinglayer.h"
#include "featuremap.h"
#include "image.h"

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
	for (int i = 0; i < trainNum; i++)
		delete pTrainImage[i];
	for (int i = 0; i < testNum; i++)
		delete pTestImage[i];
	for (int i = 0; i < layerCount; i++)
		delete mLayers[i];
	for (int i = 0; i < pFeatureMap.size(); i++)
		delete pFeatureMap[i];
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
	const string* TEST_NAME = NULL, const string* TESTLABEL_NAME = NULL)
{
	freopen((char*)pconfig_name,"r",stdin);

	scanf("%d%d%d%d",&trainNum,&testNum,&alpha,&layerCount);
	scanf("%d%d",&image_h,&image_w);
	createFeatureMap(image_h, image_w);
	for (int i = 0; i < layerCount; i++)
	{
		int type; scanf("%d", &type);
		Layer* curLayer;
		if (type == 1)curLayer = new ConvolutionLayer;
		if (type == 2)curLayer = new SubSamplingLayer;
		if (type == 3)curLayer = new FullConnectionLayer;
		mLayers.push_back(curLayer);
	}

	fclose(stdin);


	if (readData(pTrainImage, trainLabel, trainNum, TRAIN_NAME, TRAINLABEL_NAME))
		printf("can't find training data!\n"), system("pause");
	if(readData(pTestImage, testLabel, testNum, TEST_NAME, TESTLABEL_NAME))
		printf("can't find testing data!\n"), system("pause");

}

void NeuralNetwork::trainBatch(int batchsize)
{
	// ÅÐ¶ÏbatchsizeÕû³ýtrainNumÂð
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->randomize();
	for (int i = 0; i < trainNum / batchsize; i++)
	{
		for (int j = 0; j < batchsize; j++)
			train(pTrainImage[i*batchsize+j],&trainLabel[i*batchsize+j]);
		for (int j = 0; j < layerCount; j++)
			mLayers[j]->update(alpha);
	}
}

void NeuralNetwork::testBatch()
{
	int percent = 0, correct=0;
	for (int i = 0; i < testNum; i++)
	{
		if (test(pTestImage[i])==testLabel[i])correct++;
		if ((double)(i + 1) / testNum * 10 > percent)
		{
			percent = (double)(i + 1) / testNum * 10;
			printf("%d0 %%, accuracy: %llf\n", percent,(double)correct/(i+1));
		}
	}
}

FeatureMap* NeuralNetwork::getFeatureMap(){return pFeatureMap[curFeatureMap++]; }
FeatureMap* NeuralNetwork::getError(){return pError[curError++];}

FeatureMap* NeuralNetwork::createFeatureMap(int height,int width){return pFeatureMap[++featureMapCount] = new FeatureMap(height,width);}
FeatureMap* NeuralNetwork::createError(int height, int width) { return pError[++errorCount] = new FeatureMap(height, width); }

void NeuralNetwork::train(Image* image,uint8* label)
{
	image->transform(pFeatureMap[0]);
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->forward(relu);
	softmax(*label);
	for (int i = layerCount - 1; i >= 0; i--)
		mLayers[i]->backward(relugrad);
}
uint8 NeuralNetwork::getResult()
{
	int result=0;
	for (int i = curFeatureMap; i < featureMapCount; i++)
		if (pFeatureMap[result]->data[0][0] < pFeatureMap[result]->data[0][0])
			result = i;
	return result;
}

uint8 NeuralNetwork::test(Image* image)
{
	image->transform(pFeatureMap[0]);
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->forward(relu);
	return getResult();
}

bool NeuralNetwork::readData(vector<Image*> image, vector<uint8> label, int train_or_test_count, const string* cData, const string* cLabel)
{
	FILE* fp_image = fopen((char*)cData, "rb");
	FILE* fp_label = fopen((char*)cLabel, "rb");
	if (!fp_image || !fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);

	uint8* tImage = new uint8[train_or_test_count*Image::sh*Image::sw];
	fread(tImage, Image::sh * Image::sw * train_or_test_count, 1, fp_image);
	for (int i = 0; i < train_or_test_count; i++)
		for (int j = 0; j < Image::sh; j++)
			for (int k = 0; k < Image::sw; k++)
				image[i]->data[j][k] = tImage[i* Image::sh * Image::sw+j*Image::sw+k];

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
	for (uint8 i = curFeatureMap; i < featureMapCount; ++i)
	{
		pError[curError+i-curFeatureMap]->data[0][0] = exp(pFeatureMap[i]->data[0][0] - max);
		k += pError[curError + i - curFeatureMap]->data[0][0];
	}
	k = 1. / k;
	for (uint8 i = curError; i < errorCount; ++i)
	{
		pError[i]->data[0][0] *= k;
		inner -= (pError[i]->data[0][0])* (pError[i]->data[0][0]);
	}
	inner += pError[label+curError]->data[0][0];
	for (uint8 i = curError; i < errorCount; ++i)
	{
		pError[i]->data[0][0] *= (i == label+curError) - pError[i]->data[0][0] - inner;
	}
}