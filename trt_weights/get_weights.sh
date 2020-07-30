# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

# Downloads with gdown
gdown -O yolov4_1_608_608.trt https://drive.google.com/uc?id=1LlsfIf-CApt9bNLd_-HDYOBTGvy_UY-3
gdown -O yolov4_2_608_608.trt https://drive.google.com/uc?id=1fl6St0D6qTCTt-Ade_3iV5qVJXVndivk
gdown -O yolov4_4_608_608.trt https://drive.google.com/uc?id=1g2JB-SyKLNOXPI4Iq8ztnc62itR2L4I7
gdown -O yolov4_8_608_608.trt https://drive.google.com/uc?id=1nKULKPsVfphqrjicQg4gqdiegMSRpOvF
