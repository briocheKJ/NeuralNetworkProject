#pragma once
#include "layer.h"
#include<vector>
#include<string>

#define uint8 unsigned char

using namespace std;
using std::string;
using std::vector;

class Layer;
class FeatureMap;
class Image;

class NeuralNetwork
{
public:
	~NeuralNetwork();//释放内存image，featuremap，layer
	static NeuralNetwork* getInstance();
	static void releaseInstance();

public:
	void initialize(const string* pconfig_name,
		const string* TRAIN_NAME, const string* TRAINLABEL_NAME,
		const string* TEST_NAME = NULL, const string* TESTLABEL_NAME = NULL);
	//读入config的结构相关部分
	//初始化layerCount，初始化各层，记录各层指针到mLayers，下派各层读config的任务
	//初始化trainNum和testNum，读入所有的image到pTrainImage和pTestImage的所在位置。
	void trainBatch(int batchsize);//全部分批训练，batchsize表示分批的大小
	void testBatch();//全部测试
	uint8 testSingle(Image*);
public:
	FeatureMap* createFeatureMap(int pid, int height, int width);
	FeatureMap* getFeatureMap(int pid);

	FeatureMap* createError(int pid, int height, int width);
	FeatureMap* getError(int pid);

private:
	uint8 getResult(int pid);
	void softmax(int pid, uint8 label);
	void train(int pid, Image* image, uint8* label);
	uint8 test(Image* image);
	bool readData(vector<Image*> &image, vector<uint8> &label, int train_or_test_count, const string* cData, const string* cLabel);

private:
	static NeuralNetwork* spNeuralNetwork;

private:
	vector<Layer*> mLayers;
	vector<FeatureMap*> pFeatureMap[THREAD_NUM];
	vector<FeatureMap*> pError[THREAD_NUM];
	vector<Image*> pTrainImage;
	vector<Image*> pTestImage;
	vector<uint8> trainLabel;
	vector<uint8> testLabel;

private:
	double alpha;

	int curFeatureMap[THREAD_NUM] = { 0 };
	int featureMapCount[THREAD_NUM] = { 0 };
	int curError[THREAD_NUM] = { 0 };
	int errorCount[THREAD_NUM] = { 0 };

	int layerCount;
	int trainNum = 0;
	int testNum = 0;

	int image_h;
	int image_w;

	int firstH;
	int firstW;
};