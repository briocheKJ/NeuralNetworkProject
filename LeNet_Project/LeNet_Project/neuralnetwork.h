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
	~NeuralNetwork();//�ͷ��ڴ�image��featuremap��layer
	static NeuralNetwork* getInstance();
	static void releaseInstance();

public:
	void initialize(const string* pconfig_name,
		const string* TRAIN_NAME, const string* TRAINLABEL_NAME,
		const string* TEST_NAME = NULL, const string* TESTLABEL_NAME = NULL);
	//����config�Ľṹ��ز���
	//��ʼ��layerCount����ʼ�����㣬��¼����ָ�뵽mLayers�����ɸ����config������
	//��ʼ��trainNum��testNum���������е�image��pTrainImage��pTestImage������λ�á�
	void trainBatch(int batchsize);//ȫ������ѵ����batchsize��ʾ�����Ĵ�С
	void testBatch();//ȫ������
	uint8 testSingle(Image*);
public:
	FeatureMap* createFeatureMap(int height, int width);
	FeatureMap* getFeatureMap();

	FeatureMap* createError(int height, int width);
	FeatureMap* getError();

private:
	uint8 getResult();
	void softmax(uint8 label);
	void train(Image* image, uint8* label);
	uint8 test(Image* image);
	bool readData(vector<Image*> image, vector<uint8> label, int train_or_test_count, const string* cData, const string* cLabel);

private:
	static NeuralNetwork* spNeuralNetwork;

private:
	vector<Layer*> mLayers;
	vector<FeatureMap*> pFeatureMap;
	vector<FeatureMap*> pError;
	vector<Image*> pTrainImage;
	vector<Image*> pTestImage;
	vector<uint8> trainLabel;
	vector<uint8> testLabel;

private:
	int alpha;

	int curFeatureMap = 0;
	int featureMapCount = 0;
	int curError = 0;
	int errorCount = 0;

	int layerCount;
	int trainNum = 0;
	int testNum = 0;

	int image_h;
	int image_w;
};