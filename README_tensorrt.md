# Pytorch-YOLOv4 (TensorRT)

## Setup

- in tool/yolo_layer.py, change last line from yolo_forward_alternative to yolo_forward (RMB TO CHANGE BACK AFTER FINISH CONVERTING)
- start docker

## Conversion from PyTorch to ONNX

```
python3 demo_pytorch2onnx.py weights/yolov4.pth <image_path> <batch_size> 80 <IN_IMAGE_H> <IN_IMAGE_W>
```

With batch_size=1, IN_IMAGE_H=608, IN_IMAGE_W=608:
```
python3 -m onnxsim yolov4_1_3_608_608.onnx yolov4_1_3_608_608_sim.onnx

cp yolov4_1_3_608_608_sim.onnx /usr/src/tensorrt/bin
```

## Conversion from ONNX to TensorRT

```
cd /usr/src/tensorrt/bin/

./trtexec --onnx=yolov4_1_3_608_608_sim.onnx --maxBatch=1 --saveEngine=yolov4_1_3_608_608_sim.trt --fp16 --verbose

cp yolov4_1_3_608_608_sim.trt /retinaface_rose/pytorch_YOLOv4/trt_weights/
```

*NOTE:* maxBatch should be the same as onnx batch size -> if value is higher than onnx batch size, no error but inference will be wrong

## Run

``` 
cd /retinaface_rose/pytorch_YOLOv4/

python3 demo_trt.py trt_weights/yolov4_1_3_608_608_sim.trt <image_path> 608 608 <img_bs>
```

## Notes

- Right now only works for static image sizes
- If use explicitBatch in trtexec, the trt engine max_batch_size will always be 1
