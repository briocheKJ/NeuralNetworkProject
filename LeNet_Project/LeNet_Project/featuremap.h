#pragma once

class FeatureMap
{
public:
	FeatureMap(int _h, int _w); //����h*w��FeatureMap
	~FeatureMap();
	int h;
	int w;
	double** data; //�洢h*w��FeatureMap

	void clear();
}; //FeatureMap�ࣺһ������洢һ������ͼ