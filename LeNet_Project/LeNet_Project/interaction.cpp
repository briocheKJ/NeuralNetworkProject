#include <iostream>
using namespace std;
#include<time.h>
#include <string>
//#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
//#include<opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/types_c.h>
#include "interaction.h"
#include "neuralnetwork.h"
#include "image.h"
using namespace cv;
interaction* interaction::instance = 0;
Point prePoint;				//�������ָ�����һ����
Mat canvas ;			//�ڵ׻���
Mat digit ;			//���д�µ���д������



static void draw_circle(int event, int x, int y, int flags, void*) {
	
	if (event == EVENT_LBUTTONDOWN)			//�����������£���ȡ�õ�����
	{
		prePoint = Point(x, y);
	}
	//�����������ְ��±�־��������ƶ�
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		Point nowPoint(x, y);
		line(digit, nowPoint, prePoint, Scalar(255), 2, 8, 0);
		prePoint = nowPoint;		//�����ĵ�ǰ����Ϊ�ϸ���λ���Կ�ʼ��һ���߶�����Ļ���
		imshow("digit", digit);			//�ڴ��ڸ���Ŀ��ͼ��
	}
	//������Ҽ�����ʱ
	else if (event == EVENT_RBUTTONDOWN)
	{
		imshow("digit", canvas);
		digit = canvas.clone();
	}
}
void interaction::InputPicture() {
	
}
void interaction::Management() {
	canvas = Mat::zeros(Size(28, 28), CV_8UC1);
	digit = canvas.clone();
	namedWindow("digit", WINDOW_NORMAL);
	resizeWindow("digit", Size(200, 200));
	imshow("digit", canvas);
	setMouseCallback("digit", draw_circle, 0);			//�����������������λ�ڸ�ָ��������ʱ��ִ�лص�����
	while (true)
	{
		char ch = waitKey();
		if (ch == ' ')
		{

			
			break;
			
		}

		else if (ch == 27)
		{
			break;
		}

	}

	//waitKey(0);

	destroyWindow("digit");

	int row = Image::sh;
	int col = Image::sw;
	
	resize(digit, digit, Size(28, 28));
	
	Image* pimage = new Image();
	
	for(int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			pimage->data[i][j] = digit.at<uchar>(i, j);
		}
	}
	int t;
		t=(int)NeuralNetwork:: getInstance()->testSingle(pimage);
	
	 for (int i = 0; i < 28; i++)
	 {
		 for (int j = 0; j < 28; j++) {
			 if ((int)pimage->data[i][j])cout << (int)pimage->data[i][j] / 27<<" ";
			 else cout << "0";
		 }
		 cout << endl;
	 }
	 cout << "���ԵĽ���ǣ�" << t << endl;
	delete pimage;
}
