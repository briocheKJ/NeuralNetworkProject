#include "image.h"
#include "featuremap.h"
#include <cmath>

int Image::sw = 0;
int Image::sh = 0;

void Image::setwh(int h,int w) {sw = w; sh = h;}

void Image::transform(FeatureMap* featuremap,int firstH,int firstW)
{
	int sz = sw*sh;
	double mean = 0, std = 0;
	for (int j = 0; j < sh; ++j)
		for (int k = 0; k < sw; ++k)
		{
			mean += data[j][k];
			std += data[j][k] * data[j][k];
		}
	mean /= sz;
	std = sqrt(std / sz - mean * mean);
	int paddingX=(firstH-sh)>>1, paddingY = (firstW-sw)>>1;
	for (int i = 0; i < sh; i++)
		for (int j = 0; j < sw; j++)
		{
			featuremap->data[i+ paddingX][j + paddingY] = (data[i][j] - mean) / std;
		}
}

Image::Image()
{
	data = new uint8*[sh];
	for (int i = 0; i < sh; i++)
		data[i] = new uint8[sw];
}

Image::~Image()
{
	for (int i = 0; i < sh; i++)
		delete [] data[i];
	delete [] data;
}