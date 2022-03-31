#pragma once
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
		const string* ptrain_name, const string* ptrain_label_name, 
		const string* ptest_name = NULL, const string* ptest_label_name=NULL);
	//读入config的结构相关部分
	//初始化layerCount，初始化各层，记录各层指针到mLayers，下派各层读config的任务
	//初始化trainNum和testNum，读入所有的image到pTrainImage和pTestImage的所在位置。
	void trainBatch();//批量训练
	void testBatch();//批量测试
	void testSingle(Image*);//单独测试

public:
	FeatureMap* createFeatureMap(int height, int width);

private:

	void train(Image* image,uint8 *label);
	void test(Image* image, uint8* label);
	void readData(Image* image, uint8* label, int train_or_test_count, const string* cData, const string* cLabel);

private:
	static NeuralNetwork* spNeuralNetwork;

private:
	vector<Layer*> mLayers;
	vector<FeatureMap*> pFeatureMap;
	vector<Image*> pTrainImage;
	vector<Image*> pTestImage;

private:
	int layerCount;
	int featureMapCount = 0;
	int trainNum = 0;
	int testNum = 0;
};