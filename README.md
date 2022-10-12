# yolov5-seg-opencv-dnn-cpp
使用opencv-dnn部署yolov5实例分割模型
基于6.2版本的yolov5:https://github.com/ultralytics/yolov5

**OpenCV>=4.5.0**

```
python export.py --weights yolov5s-seg.pt --img [640,640] --include onnx --dnn
```
 
#### 2022.10.10 更新：  
+ 0.opencv不支持动态推理，请不要加--dymanic导出onnx。
+ 1.关于换行符，windows下面需要设置为CRLF，上传到github会自动切换成LF，windows下面切换一下即可<br>
+ 2.有些小伙伴用版本为1.12.x的pytorch的时候，需要将
https://github.com/ultralytics/yolov5/blob/c98128fe71a8676037a0605ab389c7473c743d07/export.py#L155
这里的标志位改成```do_constant_folding=False, ```,否者opencv用dnn读取不了onnx文件
+ **4.关于mask**
  > 原始的mask采用的是整张图片mask，即使你的box很小，整张图就一小块区域，也会是整个mask。修改之后变成了box内mask，mask跟着box的大小走（原始代码中的crop操作），提升速度的同时，内存开销也会减小。
  ![](res/boxMask.bmp)

以下为yolov5-seg.onnx运行结果：
![](res/bus.bmp)
![](res/zidane.bmp)