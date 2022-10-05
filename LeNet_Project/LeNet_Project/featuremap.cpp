#include "featuremap.h"
#include <cstring>

FeatureMap::FeatureMap(int _h, int _w) :h(_h), w(_w)
{
	data = new double* [h];
	for (int i = 0; i < h; i++)
	{
		data[i] = new double[w];
		for (int j = 0; j < w; j++)
			data[i][j] = 0.0;
	}
}

FeatureMap::~FeatureMap()
{
	for (int i = 0; i < h; i++)
		delete[] data[i];
	delete[] data;
}

void FeatureMap::clear()
{
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			data[i][j] = 0.0;
}