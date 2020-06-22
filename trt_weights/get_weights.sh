files=( "yolov4_1_3_608_608_sim.trt:1pJFODUP_Z7fVPGY1DYhmyK7RjMc0_T3u"
        "yolov4_2_3_608_608_sim.trt:1oHwmZiDlOx2JvfIVlCX5GUx0Bwwn_1ke" )

for file in "${files[@]}" ; do
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${file#*:} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="${file#*:} -O ${file%:*} && rm -rf /tmp/cookies.txt
done