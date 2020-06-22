# Pytorch-YOLOv4 (TensorRT)

## Conversion from PyTorch to ONNX

```
python3 demo_pytorch2onnx.py weights/yolov4.pth <image_path> <batch_size> 80 <IN_IMAGE_H> <IN_IMAGE_W>
```

With batch_size=1, IN_IMAGE_H=608, IN_IMAGE_W=608:
```
python3 -m onnxsim yolov4_1_3_608_608.onnx yolov4_1_3_608_608_sim.onnx --input-shape 1,3,608,608

cp yolov4_1_3_608_608_sim.onnx /usr/src/tensorrt/bin
```

## Conversion from ONNX to TensorRT

```
cd /usr/src/tensorrt/bin/

./trtexec --onnx=yolov4_1_3_608_608_sim.onnx --explicitBatch --saveEngine=yolov4_1_3_608_608_sim.trt --workspace=4096 --fp16 --verbose

cp yolov4_1_3_608_608_sim.trt /retinaface_rose/pytorch_YOLOv4/trt_weights
```

## Run

``` 
python3 demo_trt.py trt_weights/yolov4_1_3_608_608_sim.trt <image_path> 608 608
```


