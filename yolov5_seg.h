#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include "yolov5_utils.h"

class Yolov5Seg {
public:
	Yolov5Seg() {
	}
	~Yolov5Seg() {}
	/** \brief Read onnx-model
	* \param[out] read onnx file into cv::dnn::Net
	* \param[in] modelPath:onnx-model path
	* \param[in] isCuda:if true and opencv built with CUDA(cmake),use OpenCV-GPU,else run it on cpu.
	*/
	bool ReadModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	/** \brief  detect.
	* \param[in] srcImg:a 3-channels image.
	* \param[out] output:detection results of input image.
	*/
	bool Detect(cv::Mat& srcImg, cv::dnn::Net& net, std::vector<OutputSeg>& output);

	 int _netWidth = 640;   //ONNX图片输入宽度
	 int _netHeight = 640;  //ONNX图片输入高度
public:
	std::vector<std::string> _className = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush" };//类别名，换成自己的模型需要修改此项
private:
	float _classThreshold = 0.25;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;
};
