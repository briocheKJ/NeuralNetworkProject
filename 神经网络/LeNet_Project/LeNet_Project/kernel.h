#pragma once

class Kernel
{
public:
	Kernel(int _num, int _h, int _w); //����ʱ��w,b�ռ�
	~Kernel();
	int num; //�������������w,b��һά��
	int height; //����˸�
	int width; //����˿�
	double*** w; //weight
	double* b; //bias
}; //Kernel�ࣺһ�������һ�����˲���