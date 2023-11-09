#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>
#include "yolov5.h"
#include "yolov5_seg.h"
#include "yolov5_onnx.h"
#include "yolov5_seg_onnx.h"
#include<time.h>
#define  VIDEO_OPENCV //if define, use opencv for video.
using namespace std;
using namespace cv;
using namespace dnn;

template<typename _Tp>
int yolov5(_Tp& cls, Mat& img, string& model_path)
{
	Net net;
	if (cls.ReadModel(net, model_path, false)) {
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


	if (cls.Detect(img, net, result)) {
		DrawPred(img, result, cls._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}

template<typename _Tp>
int yolov5_onnx(_Tp& cls, Mat& img, string& model_path)
{

	if (cls.ReadModel(model_path, false)) {
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
	if (cls.OnnxDetect(img, result)) {
		DrawPred(img, result, cls._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}

template<typename _Tp>
int video_demo(_Tp& cls, string& model_path)
{
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "open capture failured!" << std::endl;
		return -1;
	}
	Mat frame;
#ifdef VIDEO_OPENCV
	Net net;
	if (cls.ReadModel(net, model_path, true)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read net failured!" << endl;
		return -1;
	}

#else
	if (cls.ReadModel(model_path, true)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read net failured!" << endl;
		return -1;
	}

#endif

	while (true)
	{

		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "read to end" << std::endl;
			break;
		}
		result.clear();
#ifdef VIDEO_OPENCV

		if (cls.Detect(frame, net, result)) {
			DrawPred(frame, result, cls._className, color, true);
		}
#else
		if (cls.OnnxDetect(frame, result)) {
			DrawPred(frame, result, cls._className, color, true);
		}
#endif
		int k = waitKey(10);
		if (k == 27) { //esc 
			break;
		}

	}
	cap.release();

	system("pause");

	return 0;
}

int main() {
	string img_path = "./images/zidane.jpg";
	string seg_model_path = "./models/yolov5s-seg.onnx";
	string detect_model_path = "./models/yolov5s.onnx";
	Mat img = imread(img_path);

	Yolov5 task_detect;
	Yolov5Seg task_segment;
	Yolov5Onnx task_detect_onnx;
	Yolov5SegOnnx task_segment_onnx;
	Mat temp = img.clone();
	yolov5(task_detect, temp, detect_model_path);    //Opencv detect
	//temp = img.clone();
	//yolov5(task_segment, temp, seg_model_path);   //opencv segment
	//temp = img.clone();
	//yolov5_onnx(task_detect_onnx, temp, detect_model_path);  //onnxruntime detect
	//temp = img.clone();
	//yolov5_onnx(task_segment_onnx, temp, seg_model_path); //onnxruntime segment
#ifdef VIDEO_OPENCV
	video_demo(task_detect, detect_model_path);
	//video_demo(task_segment, seg_model_path);
#else
	video_demo(task_detect_onnx, detect_model_path);
	//video_demo(task_segment_onnx, seg_model_path);
#endif

	return 0;
}


