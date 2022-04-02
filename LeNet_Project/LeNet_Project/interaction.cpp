#include <iostream>
using namespace std;
#include<time.h>
#include <string>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include "interaction.h"
#include "neuralnetwork.h"
#include "image.h"
using namespace cv;
interaction* interaction::instance = 0;
Mat img = Mat::zeros(Size(400, 400), CV_8UC3);
Point p1, p2;
int point_num = 0;
clock_t previous = clock(), current;
double duration;

static void draw_circle(int event, int x, int y, int flags, void*) {
	
	if ((event == EVENT_MOUSEMOVE) && (flags & EVENT_FLAG_LBUTTON)) {
		p2 = Point(x, y);
		current = clock();
		duration = (double)(current - previous);
		
		if (duration > 200) {
			point_num = 0;
		}
		
		if (point_num != 0) {

			line(img, p1, p2, Scalar(0, 0, 0));
		}
		else {
			circle(img, p2, 1, Scalar(0, 0, 0));
		}
		p1 = p2;
		previous = current;
		point_num++;
	}
}
void interaction::InputPicture() {
	
}
void interaction::Management() {
	img = Mat::zeros(Size(400, 400), CV_8UC3);
	img = Scalar(255, 255, 255);
	Scalar color = Scalar(0, 0, 255);
	namedWindow("image", WINDOW_AUTOSIZE);
	setMouseCallback("image", draw_circle);
	while (1) {
		imshow("image", img);
		if (waitKey(1) == 13) {
			break;
		}
	}

	//waitKey(0);
	destroyWindow("image");
	int row = Image::sh;
	int col = Image::sw;
	resize(img, img, Size(col, row));
	Image* pimage = new Image();
	for (int i = 1; i <= row; i++) {
		for (int j = 1; j <= col; j++) {
			pimage->data[i][j] = img.at<uchar>(i, j);
		}
	}
	// NeuralNetwork:: getInstance->testSingle(pimage);
	delete pimage;
}
