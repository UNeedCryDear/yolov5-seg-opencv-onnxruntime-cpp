#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include "yolov5_utils.h"

class Yolov5 {
public:
	Yolov5() {
	}
	~Yolov5() {}
	/** \brief Read onnx-model
	* \param[in] net:onnx-model 
	* \param[in] netPath:onnx-model path
	* \param[in] isCuda:if true,use GPU(neeed build opencv-cuda),else run it on cpu.
	*/
	bool ReadModel(cv::dnn::Net& net, std::string& netPath, bool isCuda = false);
	bool Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<OutputSeg>& output);

private:

	const int _netWidth = 640;   //ONNX图片输入宽度
	const int _netHeight = 640;  //ONNX图片输入高度

	float _classThreshold = 0.25;
	float _nmsThreshold = 0.45;
public:
	std::vector<std::string> _className = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush" };
};
