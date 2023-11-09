#include"yolov5.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;


bool Yolov5::ReadModel(Net& net, string& netPath, bool isCuda) {
	try {
		if (!CheckModelPath(netPath))
			return false;
		net = readNetFromONNX(netPath);
#if CV_VERSION_MAJOR==4 &&CV_VERSION_MINOR==7&&CV_VERSION_REVISION==0
		net.enableWinograd(false);  //bug of opencv4.7.x in AVX only platform ,https://github.com/opencv/opencv/pull/23112 and https://github.com/opencv/opencv/issues/23080 
		//net.enableWinograd(true);		//If your CPU supports AVX2, you can set it true to speed up
#endif
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}
bool Yolov5::Detect(Mat& SrcImg, Net& net, vector<OutputSeg>& output) {
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	Vec4d params;
	LetterBox(SrcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));

	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	//vector<string> outputLayerName{"345","403", "461","output" };
	//net.forward(netOutputImg, outputLayerName[3]); //获取output的输出
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	float ratio_h = (float)netInputImg.rows / _netHeight;
	float ratio_w = (float)netInputImg.cols / _netWidth;
	int net_width = _className.size() + 5;  //输出的网络宽度是类别数+5
	int net_out_width = netOutputImg[0].size[2];
	assert(net_out_width == net_width, "Error Wrong number of _className");  //模型类别数目不对
	float* pdata = (float*)netOutputImg[0].data;
	int net_height = netOutputImg[0].size[1];
	for (int r = 0; r < net_height; ++r) {  
		float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
		if (box_score >= _classThreshold) {
			cv::Mat scores(1, _className.size(), CV_32FC1, pdata + 5);
			Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = max_class_socre* box_score;
			if (max_class_socre >= _classThreshold) {
				//rect [x,y,w,h]
				float x = (pdata[0] - params[2]) / params[0];
				float y = (pdata[1] - params[3]) / params[1];
				float w = pdata[2] / params[0];
				float h = pdata[3] / params[1];
				int left = MAX(round(x - 0.5 * w ), 0);
				int top = MAX(round(y - 0.5 * h ), 0);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre );
				boxes.push_back(Rect(left, top, round(w * ratio_w), round(h * ratio_h)));
			}
		}
		pdata += net_width;//下一行

	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}
	if (output.size())
		return true;
	else
		return false;
}

