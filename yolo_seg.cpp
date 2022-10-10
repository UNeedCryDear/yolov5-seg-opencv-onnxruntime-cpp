#include"yolo_seg.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

bool YoloSeg::ReadModel(Net& net, string& netPath, bool isCuda = false) {
	try {
		net = readNet(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}
void YoloSeg::LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int newUnpad[2]{ (int)std::round((float)shape.width * r),
		(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - newUnpad[0]);
	auto dh = (float)(newShape.height - newUnpad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		newUnpad[0] = newShape.width;
		newUnpad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
	{
		cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void YoloSeg::GetMask(const Mat& maskProposals, const Mat& mask_protos, const cv::Vec4d& params, const cv::Size& srcImgShape, vector<OutputSeg>& output) {
	Mat protos = mask_protos.reshape(0, { _segChannels,_segWidth * _segHeight });
	Mat matmulRes = (maskProposals * protos).t();
	Mat masks = matmulRes.reshape(output.size(), { _segWidth,_segHeight });
	vector<Mat> maskChannels;
	split(masks, maskChannels);
	for (int i = 0; i < output.size(); ++i) {
		Mat dest, mask;
		//sigmoid
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);

		Rect roi(int(params[2] / _netWidth * _segWidth), int(params[3] / _netHeight * _segHeight), int(_segWidth - params[2] / 2), int(_segHeight - params[3] / 2));
		dest = dest(roi);
		resize(dest, mask, srcImgShape, INTER_LINEAR);
		mask = mask > _maskThreshold;
		//crop
		Mat m = Mat::zeros(srcImgShape, CV_8UC1);
		Rect temp_rect = output[i].box;
		rectangle(m, temp_rect, Scalar(255), -1, 8);
		mask &= m;

		output[i].mask = mask;

	}
}

bool YoloSeg::Detect(Mat& SrcImg, Net& net, vector<OutputSeg>& output) {
	Mat blob;
	output.clear();
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg;
	Vec4d params;
	LetterBox(SrcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	//net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	//*********************************************************************************************************************************
	//opencv4.5.x和4.6.x这里输出不一致，推荐使用下面的固定名称输出
	// 如果使用net.forward(netOutputImg, net.getUnconnectedOutLayersNames())，需要确认下output0在前，output1在后，否者出错
	//*********************************************************************************************************************************
	vector<string> outputLayerName{ "output0","output1" };
	net.forward(netOutputImg, outputLayerName); //获取output的输出

	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<vector<float>> picked_proposals;  //存储output0[:,:, 5 + _className.size():net_width]用以后续计算mask

	float ratio_h = (float)netInputImg.rows / _netHeight;
	float ratio_w = (float)netInputImg.cols / _netWidth;
	int net_width = _className.size() + 5 + _segChannels;
	float* pdata = (float*)netOutputImg[0].data;
	for (int stride = 0; stride < _strideSize; stride++) {    //stride
		int grid_x = (int)(_netWidth / _netStride[stride]);
		int grid_y = (int)(_netHeight / _netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) {	//anchors
			const float anchor_w = _netAnchors[stride][anchor * 2];
			const float anchor_h = _netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; ++i) {
				for (int j = 0; j < grid_x; ++j) {
					float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
					if (box_score >= _boxThreshold) {
						cv::Mat scores(1, _className.size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = (float)max_class_socre;
						if (max_class_socre >= _classThreshold) {

							vector<float> temp_proto(pdata + 5 + _className.size(), pdata + net_width);
							picked_proposals.push_back(temp_proto);
							//rect [x,y,w,h]
							float x = (pdata[0] - params[2]) / params[0];  //x
							float y = (pdata[1] - params[3]) / params[1];  //y
							float w = pdata[2] / params[0];  //w
							float h = pdata[3] / params[1];  //h
							int left = (x - 0.5 * w) * ratio_w;
							int top = (y - 0.5 * h) * ratio_h;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * box_score);
							boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
						}
					}
					pdata += net_width;//下一行
				}
			}
		}
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, _nmsScoreThreshold, _nmsThreshold, nms_result);
	std::vector<vector<float>> temp_mask_proposals;
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		temp_mask_proposals.push_back(picked_proposals[idx]);
		output.push_back(result);
	}
	Mat mask_proposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
		mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());

	GetMask(mask_proposals, netOutputImg[1], params, SrcImg.size(), output);
	
	if (output.size())
		return true;
	else
		return false;
}
void YoloSeg::DrawPred(Mat& img, vector<OutputSeg> result, vector<Scalar> color) {
	Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		mask.setTo(color[result[i].id], result[i].mask);
		string label = _className[result[i].id] + ":" + to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	addWeighted(img, 0.5, mask, 0.5, 0, img); //将mask加在原图上面
	imshow("1", img);
	imwrite("out.bmp", img);
	waitKey();
	//destroyAllWindows();

}
