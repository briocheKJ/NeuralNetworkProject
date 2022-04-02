#pragma once

class Kernel
{
public:
	Kernel(int _num, int _h, int _w); //����ʱ��w�ռ�
	~Kernel();
	int num; //�������������w,b��һά��
	int height; //����˸�
	int width; //����˿�
	double*** w; //weight
	double b; //bias

	void randomize();
	void update(Kernel* other, double alpha);
	void clear();
}; //Kernel�ࣺһ�������һ�����˲���