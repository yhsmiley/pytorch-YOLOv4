# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

gdown -O yolov4.weights https://drive.google.com/uc?id=1c3HsOmGkMTpxBwfhQS1-deOoN2We5TQY

gdown -O yolov4.pth https://drive.google.com/uc?id=1lXcUVGI2Ns5hJlAq98pQMV43Ds3vzh96