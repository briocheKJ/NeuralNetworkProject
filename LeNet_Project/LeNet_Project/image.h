#pragma once
#define uint8 unsigned char
class Image
{
public:
	static void setwh(int w,int h);
	Image();
	~Image();
	static int sh;
	static int sw;
	uint8 ** data; //´æ´¢h*wµÄImage
};

