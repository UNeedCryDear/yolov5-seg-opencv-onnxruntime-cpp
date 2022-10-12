#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>
#include "yolo_seg.h"
#include<time.h>
//#include"yolov5.h"

using namespace std;
using namespace cv;
using namespace dnn;

int yolov5_seg()
{
	string img_path = "./images/bus.jpg";
	string model_path = "./models/yolov5s-seg.onnx";
	YoloSeg test;
	Net net;
	if (test.ReadModel(net, model_path, true)) {
		cout << "read net ok!" << endl;
	}
	else {
		return -1;
	}
	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;
	Mat img = imread(img_path);
	clock_t t1, t2;
	if (test.Detect(img, net, result)) {
		test.DrawPred(img, result, color);
	}
	else {
		cout << "Detect Failed!"<<endl;
	}
	system("pause");
    return 0;
}
//int yolov5()
//{
//	string img_path = "./images/bus.jpg";
//	string model_path = "./models/yolov5s.onnx";
//	Yolo test;
//	Net net;
//	if (test.ReadModel(net, model_path, false)) {
//		cout << "read net ok!" << endl;
//	}
//	else {
//		return -1;
//	}
//	//生成随机颜色
//	vector<Scalar> color;
//	srand(time(0));
//	for (int i = 0; i < 80; i++) {
//		int b = rand() % 256;
//		int g = rand() % 256;
//		int r = rand() % 256;
//		color.push_back(Scalar(b, g, r));
//	}
//	vector<OutputSeg> result;
//	Mat img = imread(img_path);
//	if (test.Detect(img, net, result)) {
//		test.DrawPred(img, result, color);
//
//	}
//	else {
//		cout << "Detect Failed!" << endl;
//	}
//	system("pause");
//	return 0;
//}

int main() {
	//yolov5(); //https://github.com/UNeedCryDear/yolov5-opencv-dnn-cpp
	yolov5_seg();
	return 0;
}


