#pragma once
#define uint8 unsigned char
class FeatureMap;
class Image
{
public:
	static void setwh(int h,int w);
	void transform(FeatureMap* featuremap,int firstH,int firstW);
	Image();
	~Image();
	static int sh;
	static int sw;
	uint8 ** data; //´æ´¢h*wµÄImage
};

