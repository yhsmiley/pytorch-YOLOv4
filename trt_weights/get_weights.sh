# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

# Downloads with gdown
gdown -O yolov4_1_3_608_608_sim.trt https://drive.google.com/uc?id=1pJFODUP_Z7fVPGY1DYhmyK7RjMc0_T3u
gdown -O yolov4_2_3_608_608_sim.trt https://drive.google.com/uc?id=1oHwmZiDlOx2JvfIVlCX5GUx0Bwwn_1ke