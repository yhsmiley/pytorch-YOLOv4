import cv2
import time
import argparse
from pathlib import Path

from det2 import Det2

parser = argparse.ArgumentParser()
parser.add_argument('video_path', help='path to video')
parser.add_argument('--thresh', help='OD confidence threshold', default=0.4, type=float)
args = parser.parse_args()

assert args.thresh > 0.0

od = Det2(bgr=True, 
        weights= "weights/faster-rcnn/faster_rcnn_R_50_FPN_3x/model_final_280758.pkl",
        config= "configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        classes_path= 'configs/coco80.names',
        thresh=args.thresh,
        )

if args.video_path.isdigit():
    vp = int(args.video_path)
else:
    vp = Path(args.video_path)
    assert vp.is_file(),'{} not a file'.format(vp)
    vp = str(vp)
cap = cv2.VideoCapture(vp)
assert cap.isOpened(),'Cannot open video file {}'.format(vp)

cv2.namedWindow('Faster-RCNN FPN', cv2.WINDOW_NORMAL)

while True:
    # Decode
    ret, frame = cap.read()
    if not ret:
        break
    # Inference
    tic = time.perf_counter()
    dets = od.detect_get_box_in([frame], box_format='ltrb', classes=['person'])
    toc = time.perf_counter()
    print('infer duration: {:0.3f}s'.format(toc-tic))
    dets = dets[0]

    # Drawing
    show_frame = frame.copy()
    for det in dets:
        ltrb, conf, clsname = det
        l,t,r,b = ltrb
        cv2.rectangle(show_frame, (int(l),int(t)),(int(r),int(b)), (255,255,0))
        # cv2.putText(show_frame, '{}:{:0.2f}'.format(clsname, conf), (l,b), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,255,0), lineType=2)

    cv2.imshow('Faster-RCNN FPN', show_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

 