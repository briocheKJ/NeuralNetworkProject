#pragma once
#define uint8 unsigned char
class FeatureMap;
class Image
{
public:
	static void setwh(int w,int h);
	void transform(FeatureMap* featuremap);
	Image();
	~Image();
	static int sh;
	static int sw;
	uint8 ** data; //´æ´¢h*wµÄImage
};

