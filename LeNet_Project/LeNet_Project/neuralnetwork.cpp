#include <cstdio>
#include <iostream>
#include <fstream>
#include <ctime>
#include <omp.h>
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
	
	for (int i = 0; i < trainNum; i++)
		delete pTrainImage[i];
	for (int i = 0; i < testNum; i++)
		delete pTestImage[i];
	//要释放吧？

	for (int i = 0; i < layerCount; i++)
		delete mLayers[i];
	for (int i = 0; i < THREAD_NUM; i++)
		for (int j = 0; j < pFeatureMap[i].size(); j++)
			delete pFeatureMap[i][j];
	for (int i = 0; i < THREAD_NUM; i++)
		for (int j = 0; j < pError[i].size(); j++)
			delete pError[i][j];
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

	for (int i = 0; i < THREAD_NUM; i++)
	{
		createFeatureMap(i, firstH, firstW);
		createError(i, firstH, firstW);
	}

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
		printf("can't find training data!\n"), system("pause"),exit(0);
	if (TEST_NAME==nullptr|| TESTLABEL_NAME==nullptr)return;
	if(readData(pTestImage, testLabel, testNum, TEST_NAME, TESTLABEL_NAME))
		printf("can't find testing data!\n"), system("pause"),exit(0);
	
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
	// todo：判断batchsize整除trainNum吗
	//return;
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->randomize();
	int last = 0;

	clock_t start, finish;
	start = clock();

	for (int i = 0; i < trainNum / batchsize; i++)
	{
		int now = i * 100 / (trainNum / batchsize);
		if (now > last)
		{
			system("cls");
			cout << "Progress: " << now << "%" << endl;
			finish = clock();
			cout << "Time: " << (double)(finish - start) / (double)CLOCKS_PER_SEC << "s" << endl;
			last = now;
		}

#pragma omp parallel for num_threads(THREAD_NUM)
		for (int j = 0; j < batchsize; j++)
		{
			int pid=omp_get_thread_num();

			//cout << pid << endl;
			train(pid, pTrainImage[i * batchsize + j], &trainLabel[i * batchsize + j]);
			
			#pragma omp critical
			{
				for (int l = 0; l < layerCount; l++)
					mLayers[l]->updateBuffer(pid);
			}
			//更新Buffer
		}

		for (int j = 0; j < layerCount; j++)
			mLayers[j]->update(alpha / (double)batchsize);
	}
}

void NeuralNetwork::testBatch()
{
	//return;
	int percent = 0, correct=0;
	for (int i = 0; i < testNum; i++)
	{
		if (test(pTestImage[i]) ==testLabel[i])correct++;
		if ((int)((double)(i + 1) / testNum * 10) > percent)
		{
			percent = (double)(i + 1) / testNum * 10;
			printf("%d0 %%, accuracy: %llf\n", percent,(double)correct/(i+1));
		}
	}
}

FeatureMap* NeuralNetwork::getFeatureMap(int pid) { return pFeatureMap[pid][curFeatureMap[pid]++]; }
FeatureMap* NeuralNetwork::getError(int pid) { return pError[pid][curError[pid]++]; }

FeatureMap* NeuralNetwork::createFeatureMap(int pid, int height, int width) 
{ 
	++featureMapCount[pid];
	pFeatureMap[pid].push_back(new FeatureMap(height, width));
	return pFeatureMap[pid][featureMapCount[pid] - 1];
}
FeatureMap* NeuralNetwork::createError(int pid, int height, int width) 
{ 
	++errorCount[pid];
	pError[pid].push_back(new FeatureMap(height, width));
	return pError[pid][errorCount[pid] - 1];
}

void NeuralNetwork::train(int pid, Image* image, uint8* label)
{
	for (int i = 0; i < featureMapCount[pid]; i++)
		pFeatureMap[pid][i]->clear();
	for (int i = 0; i < errorCount[pid]; i++)
		pError[pid][i]->clear();
	image->transform(pFeatureMap[pid][0], firstH, firstW);
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->forward(pid, relu);
	softmax(pid, *label);
	for (int i = layerCount - 1; i >= 0; i--)
		mLayers[i]->backward(pid, relugrad);
}
uint8 NeuralNetwork::getResult(int pid)
{
	int result = curFeatureMap[pid];
	for (int i = curFeatureMap[pid]; i < featureMapCount[pid]; i++)
	{
		//cout << pFeatureMap[i]->data[0][0] << ' ';
		if (pFeatureMap[pid][result]->data[0][0] < pFeatureMap[pid][i]->data[0][0])
			result = i;
	}
	//cout << result - curFeatureMap << endl;
	return result - curFeatureMap[pid];
}

uint8 NeuralNetwork::test(Image* image)
{
	for (int i = 0; i < featureMapCount[0]; i++)
		pFeatureMap[0][i]->clear();
	for (int i = 0; i < errorCount[0]; i++)
		pError[0][i]->clear();
	image->transform(pFeatureMap[0][0], firstH, firstW);
	for (int i = 0; i < layerCount; i++)
		mLayers[i]->forward(0, relu);
	return getResult(0);
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

void NeuralNetwork::softmax(int pid, uint8 label)
{
	double max =pFeatureMap[pid][curFeatureMap[pid] + getResult(pid)]->data[0][0];
	double k = 0, inner = 0;

	for (int i = curFeatureMap[pid]; i < featureMapCount[pid]; ++i)
	{
		pError[pid][curError[pid]+i-curFeatureMap[pid]]->data[0][0] = exp(pFeatureMap[pid][i]->data[0][0] - max);
		k += pError[pid][curError[pid] + i - curFeatureMap[pid]]->data[0][0];
	}
	k = 1. / k;
	for (int i = curError[pid]; i < errorCount[pid]; ++i)
	{
		pError[pid][i]->data[0][0] *= k;
		inner -= (pError[pid][i]->data[0][0])* (pError[pid][i]->data[0][0]);
	}
	inner += pError[pid][label+curError[pid]]->data[0][0];
	for (int i = curError[pid]; i < errorCount[pid]; ++i)
	{
		pError[pid][i]->data[0][0] *= (i == label+curError[pid]) - pError[pid][i]->data[0][0] - inner;
	}
}
uint8 NeuralNetwork::testSingle(Image* image)
{
	return test(image);
}
