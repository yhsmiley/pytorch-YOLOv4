xhost +local:docker
docker run -it --gpus all -v /media/data/pytorch_YOLOv4:/pytorch_YOLOv4 --net=host --ipc host -e DISPLAY=unix$DISPLAY -e QT_X11_NO_MITSHM=1 -v /dev/:/dev/ yolov4