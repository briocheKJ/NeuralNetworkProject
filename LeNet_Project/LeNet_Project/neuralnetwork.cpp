#include <cstdio>

#include "neuralnetwork.h"
#include "convolutionlayer.h"
#include "fullconnectionlayer.h"
#include "subsamplinglayer.h"
#include "outputlayer.h"
#include "featuremap.h"
#include "image.h"



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

	scanf("%d%d%d",&trainNum,&testNum,&layerCount);
	scanf("%d%d",&image_h,&image_w);
	createFeatureMap(image_h, image_w);
	for (int i = 0; i < layerCount; i++)
	{
		int type; scanf("%d", &type);
		Layer* curLayer;
		if (type == 1)curLayer = new ConvolutionLayer;
		if (type == 2)curLayer = new SubSamplingLayer;
		if (type == 3)curLayer = new FullConnectionLayer;
		if (type == 4)curLayer = new OutPutLayer;
		mLayers.push_back(curLayer);
	}

	fclose(stdin);


	if (readData(pTrainImage[i], trainLabel, trainNum, TRAIN_NAME, TRAINLABEL_NAME))
		printf("can't find training data!\n"), system("pause");
	if(readData(pTestImage[i], testLabel, testNum, TEST_NAME, TESTLABEL_NAME))
		printf("can't find testing data!\n"), system("pause");

}

void NeuralNetwork::trainBatch(int batchsize)
{
	// ÅÐ¶ÏbatchsizeÕû³ýtrainNumÂð
	for (int i = 0; i < trainNum / batchsize; i++)
	{
		for (int j = 0; j < batchsize; j++)
			train(pTrainImage[i*batchsize+j],&trainLabel[i*batchsize+j]);
		for (int j = 0; j < layerCount; j++)
			mLayers[i]->update();
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

void NeuralNetwork::train(Image* image,uint8* lable)
{
	image->transform(pFeatureMap[0]);
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->forward();
	softmax();
	for (int i = layerCount - 1; i >= 0; i--)
		mLayers[i]->backward();
}
uint8 getResult()
{
	int result=0;
	for (int i = 1; i < OutPutN; i++)
		if (outputs[result].data[0][0] < outputs[i].data[0][0])
			result = i;
	return result;
}

uint8 NeuralNetwork::test(Image* image)
{
	image->transform(pFeatureMap[0]);
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->forward();
	return getResult();
}

bool readData(Image* image, vector<uint8> label, int train_or_test_count, const string* cData, const string* cLabel)
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
				image[i].data[j][k] = tImage[i* Image::sh * Image::sw+j*Image::sw+k];

	uint8* tplabel = new uint8[train_or_test_count];
	fread(tplabel, train_or_test_count, 1, fp_label);
	for (int i = 0; i < train_or_test_count;i++)
		label.push_back(tplabel[i]);

	fclose(fp_image);
	fclose(fp_label);
	return 0;
}
void softmax()
{
	;
}