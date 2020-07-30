# Pytorch-YOLOv4 (TensorRT)

## Setup

- get tensorrt 7.0 for ubuntu 18.04, cuda 10.0 using ./get_trt.sh
- build docker with tensorrt.Dockerfile
- start docker with ./start_docker.sh

## Conversion from PyTorch to ONNX

```
python3 demo_pytorch2onnx.py weights/yolov4.pth <image_path> <batch_size> 80 <IN_IMAGE_H> <IN_IMAGE_W>
```

- Optionally, use onnxsim to simplify the onnx model (it works even without simplification, putting it here for future reference)<br>

With batch_size=1, IN_IMAGE_H=608, IN_IMAGE_W=608:
```
python3 -m onnxsim yolov4_1_608_608.onnx yolov4_1_608_608.onnx
```

```
cp yolov4_1_608_608.onnx /usr/src/tensorrt/bin
```

## Conversion from ONNX to TensorRT

```
cd /usr/src/tensorrt/bin/

./trtexec --onnx=yolov4_1_608_608.onnx --maxBatch=1 --saveEngine=yolov4_1_608_608.trt --fp16 --verbose

cp yolov4_1_608_608.trt /pytorch_YOLOv4/trt_weights/
```

*NOTE:* maxBatch should be the same as onnx batch size -> if value is higher than onnx batch size, no error but inference will be wrong

## Run

``` 
cd /pytorch_YOLOv4/

python3 demo_trt.py trt_weights/yolov4_1_608_608.trt <image_path> 608 608 <img_bs>
```

## Notes

- Right now only works for static image sizes
- If use explicitBatch in trtexec, the trt engine max_batch_size will always be 1
