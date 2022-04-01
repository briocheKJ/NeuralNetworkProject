#include "kernel.h"

Kernel::Kernel(int _num, int _h, int _w) :num(_num), height(_h), width(_w)
{
	w = new double** [num];
	for (int i = 0; i < num; i++)
	{
		w[i] = new double* [height];
		for (int j = 0; j < height; j++)
			w[i][j] = new double[width];
	}
	b = new double[num];

	//参数应该随机，但我现在懒得写了
}

Kernel::~Kernel()
{
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < height; j++)
			delete[] w[i][j];
		delete[] w[i];
	}
	delete[] w;
	delete[] b;
}