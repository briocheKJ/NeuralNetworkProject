#pragma once

class Kernel
{
public:
	Kernel(int _num, int _h, int _w); //构造时开w空间
	~Kernel();
	int num; //本层输入个数（w,b第一维）
	int height; //卷积核高
	int width; //卷积核宽
	double*** w; //weight
	double b; //bias

	void randomize(int inputN, int outputN);
	void update(Kernel* other, double alpha);
	void clear();
}; //Kernel类：一个对象存一类卷积核参数