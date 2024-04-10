#include "yolov5_onnx.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;




bool Yolov5Onnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
	if (_batchSize < 1) _batchSize = 1;
	try
	{
		if (!CheckModelPath(modelPath))
			return false;
		std::vector<std::string> available_providers = GetAvailableProviders();
		auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
		if (isCuda && (cuda_available == available_providers.end()))
		{
			std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}
		else if (isCuda && (cuda_available != available_providers.end()))
		{
			//if Error code:LNK2019 of OrtSessionOptionsAppendExecutionProvider_CUDA or AppendExecutionProvider_CUDA,your onnxruntime is CPU based.
			// comment it out and rebuild. 
#if ORT_API_VERSION < ORT_OLD_VISON
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = cudaID;
			_OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
#else
			_OrtStatus = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID); 
#endif
		}
		else
		{
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}

		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
		std::wstring model_path(modelPath.begin(), modelPath.end());
		_OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
		_OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif


		//init input
		_inputNodesNum = _OrtSession->GetInputCount();

#if ORT_API_VERSION < ORT_OLD_VISON
		_inputName = _OrtSession->GetInputName(0, _OrtAllocator);
		_inputNodeNames.push_back(_inputName);
#else
		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, _OrtAllocator));
		_inputNodeNames.push_back(_inputName.get());
#endif

		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		_inputNodeDataType = input_tensor_info.GetElementType();
		_inputTensorShape = input_tensor_info.GetShape();

		if (_inputTensorShape[0] == -1)
		{
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;

		}
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
		//init output
		_outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_output_name0 = _OrtSession->GetOutputName(0, _OrtAllocator);

#else
		_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, _OrtAllocator));

#endif
		Ort::TypeInfo type_info_output0(nullptr);
		Ort::TypeInfo type_info_output1(nullptr);


		type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

#if ORT_API_VERSION < ORT_OLD_VISON
		_outputNodeNames.push_back(_output_name0);

#else
		_outputNodeNames.push_back(_output_name0.get());

#endif

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();
		
		if (isCuda && warmUp) {
			//draw run
			cout << "Start warming up" << endl;
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			input_tensors.push_back(Ort::Value::CreateTensor<float>(
				_OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
				_inputTensorShape.size()));
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(_OrtRunOptions,
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			}

			delete[]temp;
		}
	}
	catch (const std::exception&) {
		return false;
	}
	return true;

}

int Yolov5Onnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
	outSrcImgs.clear();
	Size input_size = Size(_netWidth, _netHeight);
	for (int i = 0; i < srcImgs.size(); ++i) {
		Mat temp_img = srcImgs[i];
		Vec4d temp_param = { 1,1,0,0 };
		if (temp_img.size() != input_size) {
			Mat borderImg;
			LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
			//cout << borderImg.size() << endl;
			outSrcImgs.push_back(borderImg);
			params.push_back(temp_param);
		}
		else {
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}

	int lack_num = _batchSize - srcImgs.size();
	if (lack_num > 0) {
		for (int i = 0; i < lack_num; ++i) {
			Mat temp_img = Mat::zeros(input_size, CV_8UC3);
			Vec4d temp_param = { 1,1,0,0 };
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}
	return 0;

}
bool Yolov5Onnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputSeg>& output) {
	vector<cv::Mat> input_data = { srcImg };
	std::vector<std::vector<OutputSeg>> tenp_output;
	if (OnnxBatchDetect(input_data, tenp_output)) {
		output = tenp_output[0];
		return true;
	}
	else return false;
}
bool Yolov5Onnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSeg>>& output) {
	vector<Vec4d> params;
	vector<Mat> input_images;
	Size input_size(_netWidth, _netHeight);
	//preprocessing
	Preprocessing(srcImgs, input_images, params);
	Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

	size_t input_tensor_length = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> input_tensors;
	std::vector<Ort::Value> output_tensors;
	input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

	output_tensors = _OrtSession->Run(_OrtRunOptions,
		_inputNodeNames.data(),
		input_tensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size()
	);

	//post-process
	float* pdata = output_tensors[0].GetTensorMutableData<float>();
	_outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

	int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
	int net_width = _outputTensorShape[2];
	int net_height = _outputTensorShape[1];
	int score_length = net_width - 5;
	for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
		std::vector<int> class_ids;//结果id数组
		std::vector<float> confidences;//结果每个id对应置信度数组
		std::vector<cv::Rect> boxes;//每个id矩形框
		for (int r = 0; r < net_height; r++) {    //stride
			float box_score = pdata[4]; ;//box-confidence
			if (box_score >= _classThreshold) {
				cv::Mat scores(1, score_length, CV_32FC1, pdata + 5);
				Point classIdPoint;
				double max_class_socre;
				minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
				max_class_socre = (float)max_class_socre;
				if (max_class_socre >= _classThreshold) {
					//rect [x,y,w,h]
					float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
					float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
					float w = pdata[2] / params[img_index][0];  //w
					float h = pdata[3] / params[img_index][1];  //h
					int left = MAX(int(x - 0.5 * w + 0.5), 0);
					int top = MAX(int(y - 0.5 * h + 0.5), 0);
					class_ids.push_back(classIdPoint.x);
					confidences.push_back(max_class_socre * box_score);
					boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
				}
			}
			pdata += net_width;//下一行
		}
		vector<int> nms_result;
		cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
		std::vector<vector<float>> temp_mask_proposals;
		Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows);
		std::vector<OutputSeg > temp_output;
		for (int i = 0; i < nms_result.size(); ++i) {

			int idx = nms_result[i];
			OutputSeg result;
			result.id = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx] & holeImgRect;
			temp_output.push_back(result);
		}

		output.push_back(temp_output);

	}

	if (output.size())
		return true;
	else
		return false;

	return true;
}