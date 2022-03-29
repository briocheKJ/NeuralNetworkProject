#pragma once

class FeatureMap
{
public:
	FeatureMap(int _h, int _w); //构造h*w的FeatureMap
	~FeatureMap();
	int h;
	int w;
	double** data; //存储h*w的FeatureMap
}; //FeatureMap类：一个对象存储一个特征图