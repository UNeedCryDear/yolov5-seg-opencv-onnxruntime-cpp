CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
project(YOLOv5)


SET (OpenCV_DIR path/to/opencv/build)  #opencv root
SET (ONNXRUNTIME_DIR path/to/onnxruntime)

FIND_PACKAGE(OpenCV REQUIRED)
#include_directories("")
ADD_EXECUTABLE(YOLOv5 yolov5.h yolov5_onnx.h yolov5_seg.h yolov5_seg_onnx.h yolov5_utils.h
			 main.cpp yolov5.cpp yolov5_onnx.cpp yolov5_seg.cpp yolov5_seg_onnx.cpp yolov5_utils.cpp)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

TARGET_INCLUDE_DIRECTORIES(YOLOv5 PRIVATE "${ONNXRUNTIME_DIR}/include")

TARGET_COMPILE_FEATURES(YOLOv5 PRIVATE cxx_std_14)
TARGET_LINK_LIBRARIES(YOLOv5 ${OpenCV_LIBS})

if (WIN32)
    TARGET_LINK_LIBRARIES(YOLOv5 "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    TARGET_LINK_LIBRARIES(YOLOv5 "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)