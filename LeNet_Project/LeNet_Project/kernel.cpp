#include "kernel.h"
#include <cstdlib>

Kernel::Kernel(int _num, int _h, int _w) :num(_num), height(_h), width(_w)
{
	w = new double** [num];
	for (int i = 0; i < num; i++)
	{
		w[i] = new double* [height];
		for (int j = 0; j < height; j++)
			w[i][j] = new double[width];
	}
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
}

void Kernel::randomize(int inputN, int outputN)
{
	b = 0.0;
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < height; j++)
			for (int k = 0; k < width; k++)
			{
				w[i][j][k] = rand() * (2. / RAND_MAX) - 1;
				w[i][j][k] *= sqrt(6.0 / (double)(height * width * (inputN + outputN)));
			}
	}
}

void Kernel::update(Kernel* other, double alpha)
{
	b += alpha * other->b;
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < height; j++)
			for (int k = 0; k < width; k++)
				w[i][j][k] += alpha * other->w[i][j][k];
	}
}

void Kernel::clear()
{
	b = 0.0;
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < height; j++)
			for (int k = 0; k < width; k++)
				w[i][j][k] = 0.0;
	}
}
