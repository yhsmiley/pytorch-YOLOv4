# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

# Downloads with gdown
gdown -O yolov4_1_3_608_608_sim.trt https://drive.google.com/uc?id=1SJqzVITnUH4h47woqnlE95EQTmAWmvsM
# gdown -O yolov4_2_3_608_608_sim.trt https://drive.google.com/uc?id=1JI3WH0hR2NGwp0zmlRaloEBiIMMKd-XD
# gdown -O yolov4_4_3_608_608_sim.trt https://drive.google.com/uc?id=1GOrqh7Gzooj4vETA4UN5X84ck4d8gO7s
# gdown -O yolov4_8_3_608_608_sim.trt https://drive.google.com/uc?id=1P_QauniER0jzm6E_WWd2VKpqbVX7U7Xs
