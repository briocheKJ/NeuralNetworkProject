#include "image.h"
#include "featuremap.h"

int Image::sw = 0;
int Image::sh = 0;

void Image::setwh(int w, int h) {sw = w; sh = h;}

void Image::transform(FeatureMap* featuremap)
{
	for (int i = 0; i < sh; i++)
		for (int j = 0; j < sw; j++)
			featuremap->data[i][j] = data[i][j];
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