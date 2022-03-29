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
	~NeuralNetwork();//释放内存
	static NeuralNetwork* getInstance();
	static void releaseInstance();

public:
	void initialize(const string* pconfig_name, const string* ptrain_name, const string* ptest_name = NULL);
	//初始化layerCount，记录各层指针到mLayers，初始化trainNum和testNum
	void trainBatch();
	void testBatch();
	FeatureMap* createFeatureMap(int height, int width);
private:

	void train();
	void test();
	void readData(Image* image, uint8* label, int train_or_test_count, const string* cData, const string* cLabel);


private:
	static NeuralNetwork* spNeuralNetwork;

private:
	vector<Layer*> mLayers;
	vector<FeatureMap*> featureMap;

private:
	int layerCount;
	int featureMapCount = 0;
	int trainNum;
	int testNum;
};