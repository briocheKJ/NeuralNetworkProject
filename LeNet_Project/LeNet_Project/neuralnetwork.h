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
		const string* ptrain_name, const string* ptrain_label_name, 
		const string* ptest_name = NULL, const string* ptest_label_name=NULL);
	//����config�Ľṹ��ز���
	//��ʼ��layerCount����ʼ�����㣬��¼����ָ�뵽mLayers�����ɸ����config������
	//��ʼ��trainNum��testNum���������е�image��pTrainImage��pTestImage������λ�á�
	void trainBatch();//����ѵ��
	void testBatch();//��������
	void testSingle(Image*);//��������

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