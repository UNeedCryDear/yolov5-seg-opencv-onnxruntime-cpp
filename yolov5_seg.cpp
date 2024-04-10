#include"yolov5_seg.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

bool Yolov5Seg::ReadModel(Net& net, string& netPath, bool isCuda = false) {
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
	if (isCuda) {
		//cuda
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); //or DNN_TARGET_CUDA_FP16
	}
	else {
		//cpu
		cout << "Inference device: CPU" << endl;
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}


bool Yolov5Seg::Detect(Mat& srcImg, Net& net, vector<OutputSeg>& output) {
	Mat blob;
	output.clear();
	int col = srcImg.cols;
	int row = srcImg.rows;
	Mat netInputImg;
	Vec4d params;
	LetterBox(srcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//**************************************************************************************************************************************************/
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	// If there is no problem with other settings, but results are a lot different from  Python-onnx , you can try to use the following two sentences
	// 
	//$ blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//$ blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	//****************************************************************************************************************************************************/
	net.setInput(blob);
	std::vector<cv::Mat> net_output_img;
	//*********************************************************************************************************************************
	//net.forward(net_output_img, net.getUnconnectedOutLayersNames());
	//opencv4.5.x和4.6.x这里输出不一致，推荐使用下面的固定名称输出
	// 如果使用net.forward(net_output_img, net.getUnconnectedOutLayersNames())，需要确认下net.getUnconnectedOutLayersNames()返回值中output0在前，output1在后，否者出错
	//
	// The outputs of opencv4.5.x and 4.6.x are inconsistent.Please make sure "output0" is in front of "output1" if you use net.forward(net_output_img, net.getUnconnectedOutLayersNames())
	//*********************************************************************************************************************************
	vector<string> output_layer_names{ "output0","output1" };
	net.forward(net_output_img, output_layer_names); //获取output的输出

	std::vector<int> class_ids;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
	int net_width = net_output_img[0].size[2];
	int net_height = net_output_img[0].size[1];
	int score_length = net_width - 37;
	float* pdata = (float*)net_output_img[0].data;
	for (int r = 0; r < net_height; r++) {    //lines
		float box_score = pdata[4];
		if (box_score >= _classThreshold) {
			cv::Mat scores(1, score_length, CV_32FC1, pdata + 5);
			Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			if (max_class_socre >= _classThreshold) {

				vector<float> temp_proto(pdata + 5 + score_length, pdata + net_width);
				picked_proposals.push_back(temp_proto);
				//rect [x,y,w,h]
				float x = (pdata[0] - params[2]) / params[0];  //x
				float y = (pdata[1] - params[3]) / params[1];  //y
				float w = pdata[2] / params[0];  //w
				float h = pdata[3] / params[1];  //h
				int left = MAX(int(x - 0.5 * w + 0.5), 0);
				int top = MAX(int(y - 0.5 * h + 0.5), 0);
				class_ids.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre * box_score);
				boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
		}
		pdata += net_width;//下一行

	}

	//NMS
	vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
	std::vector<vector<float>> temp_mask_proposals;
	Rect holeImgRect(0, 0, srcImg.cols, srcImg.rows);
	for (int i = 0; i < nms_result.size(); ++i) {

		int idx = nms_result[i];
		OutputSeg result;
		result.id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		temp_mask_proposals.push_back(picked_proposals[idx]);
		output.push_back(result);
	}

	MaskParams mask_params;
	mask_params.params = params;
	mask_params.srcImgShape = srcImg.size();
	mask_params.maskThreshold = _maskThreshold;
	mask_params.netHeight = _netWidth;
	mask_params.netWidth = _netWidth;
	for (int i = 0; i < temp_mask_proposals.size(); ++i) {
		GetMask2(Mat(temp_mask_proposals[i]).t(), net_output_img[1], output[i], mask_params);
	}


	//******************** ****************
	// 老版本的方案，如果上面GetMask2出错，建议使用这个。
	// If the GetMask2() still reports errors , it is recommended to use GetMask().
	//Mat mask_proposals;
	//for (int i = 0; i < temp_mask_proposals.size(); ++i) {
	//	mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());
	//}
	//GetMask(mask_proposals, net_output_img[1], output, mask_params);
	//*****************************************************/


	if (output.size())
		return true;
	else
		return false;
}

