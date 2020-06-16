import cv2
from time import perf_counter
import numpy as np
import torch
import copy

import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()
if CWD == THIS_DIR:
    from models_slower import Yolov4
else:
    from pytorch_YOLOv4.models_slower import Yolov4


class YOLOV4(object):
    if CWD == THIS_DIR:
        _defaults = {
            "weights": "weights/yolov4.pth",
            "config": "cfg/yolov4.cfg",
            "classes_path": 'data/coco.names',
            "thresh": 0.5,
            "nms_thresh": 0.4,
            "model_image_size": (608,608),
            "max_batch_size": 4,
            "half": True
        }
    else:
        _defaults = {
            "weights": "pytorch_YOLOv4/weights/yolov4.pth",
            "config": "pytorch_YOLOv4/cfg/yolov4.cfg",
            "classes_path": 'pytorch_YOLOv4/data/coco.names',
            "thresh": 0.5,
            "nms_thresh": 0.4,
            "model_image_size": (608,608),
            "max_batch_size": 4,
            "half": True
        }

    def __init__(self, bgr=True, gpu_device=0, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        # for portability between keras-yolo3/yolo.py and this
        if 'model_path' in kwargs:
            kwargs['weights'] = kwargs['model_path']
        if 'score' in kwargs:
            kwargs['thresh'] = kwargs['score']
        self.__dict__.update(kwargs) # update with user overrides

        self.class_names = self._get_class()
        # self.model = Darknet(self.config)
        # self.model.load_weights(self.weights)
        self.model = Yolov4(n_classes=len(self.class_names), inference=True)
        checkpoint = torch.load(self.weights, map_location=torch.device('cpu'))
        # checkpoint = self._rename_checkpoint(checkpoint)
        self.model.load_state_dict(checkpoint)

        self.device = gpu_device
        self.model.cuda(self.device)
        self.model.eval()

        self.bgr = bgr

        if self.half:
            self.model.half()

        # warm up
        self._detect([np.zeros((10,10,3), dtype=np.uint8)])
        print('Warmed up!')

    def _get_class(self):
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _rename_checkpoint(self, checkpoint):
        checkpoint_copy = copy.deepcopy(checkpoint)
        for key in checkpoint:
            if 'neek' in key:
                key_new = key.replace('neek', 'neck')
                checkpoint_copy[key_new] = checkpoint_copy.pop(key)
        return checkpoint_copy

    def _detect(self, list_of_imgs):
        inputs = []
        for img in list_of_imgs:
            if self.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # print('bgr: {}'.format(img.shape))
            # print('size: {}'.format(self.model_image_size))
            image = cv2.resize(img, self.model_image_size)
            # print('image: {}'.format(image.shape))
            inputs.append(np.expand_dims(np.array(image), axis=0))

        images = np.concatenate(inputs, 0)

        # print('images: {}'.format(images.shape))
        images = torch.from_numpy(images.transpose(0, 3, 1, 2)).float().div(255.0)

        images = images.cuda()
        images = torch.autograd.Variable(images)

        if self.half:
            images = images.half()

        batches = []
        for i in range(0, len(images), self.max_batch_size):
            these_imgs = images[i:i+self.max_batch_size]
            batches.append(these_imgs)

        feature_list = []
        with torch.no_grad():
            for batch in batches:
                img = batch.cuda(self.device)
                features = self.model(img)

                # feature_list = [(t, t, t), (t, t, t), (t, t, t)]
                # features = [(t, t, t), (t, t, t), (t, t, t)]
                if not feature_list:
                    feature_list = features
                else:
                    def feat_cat(feat_list_tup, feat_tup):
                        return map(lambda x, y: torch.cat((x, y)), feat_list_tup, feat_tup)

                    feature_list = map(lambda x, y: feat_cat(x, y), feature_list, features)

        output = [[feat.data.cpu().numpy() for feat in feature] for feature in feature_list]

        return self._post_processing(images, output)


    def detect_get_box_in(self, images, box_format='ltrb', classes=None, buffer_ratio=0.0):
        '''
        Params
        ------
        - images : ndarray-like or list of ndarray-like
        - box_format : string of characters representing format order, where l = left, t = top, r = right, b = bottom, w = width and h = height
        - classes : list of string, classes to focus on
        - buffer : float, proportion of buffer around the width and height of the bounding box

        Returns
        -------
        if one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
        
        else if a list of ndarray given, this return a list (batch) containing the former as the elements,

        where,
            - box_infos : list of floats in the given box format
            - score : float, confidence level of prediction
            - predicted_class : string

        '''
        single = False
        if isinstance(images, list):
            if len(images) <= 0 : 
                return None
            else:
                assert all(isinstance(im, np.ndarray) for im in images)
        elif isinstance(images, np.ndarray):
            images = [ images ]
            single = True

        res = self._detect(images)
        frame_shapes = [image.shape for image in images]
        all_dets = self._postprocess(res, shapes=frame_shapes, box_format=box_format, classes=classes, buffer_ratio=buffer_ratio)

        if single:
            return all_dets[0]
        else:
            return all_dets

    def get_detections_dict(self, frames, classes=None, buffer_ratio=0.0):
        '''
        Params: frames, list of ndarray-like
        Returns: detections, list of dict, whose key: label, confidence, t, l, w, h
        '''
        if frames is None or len(frames) == 0:
            return None
        all_dets = self.detect_get_box_in( frames, box_format='tlbrwh', classes=classes, buffer_ratio=buffer_ratio )
        
        all_detections = []
        for dets in all_dets:
            detections = []
            for tlbrwh,confidence,label in dets:
                top, left, bot, right, width, height = tlbrwh
                detections.append( {'label':label,'confidence':confidence,'t':top,'l':left,'b':bot,'r':right,'w':width,'h':height} ) 
            all_detections.append(detections)
        return all_detections

    def _postprocess(self, boxes, shapes, box_format='ltrb', classes=None, buffer_ratio=0.0):
        detections = []

        for i, frame_bbs in enumerate(boxes):
            im_height, im_width, _ = shapes[i]
            frame_dets = []
            for box in frame_bbs:
                cls_conf = box[5]
                cls_id = box[6]
                cls_name = self.class_names[cls_id]

                if classes is not None and cls_name not in classes:
                    continue

                left = int((box[0] - box[2] / 2.0) * im_width)
                top = int((box[1] - box[3] / 2.0) * im_height)
                right = int((box[0] + box[2] / 2.0) * im_width)
                bottom = int((box[1] + box[3] / 2.0) * im_height)
                w = int(right - left)
                h = int(bottom - top)
                
                width = right - left + 1
                height = bottom - top + 1
                width_buffer = width * buffer_ratio
                height_buffer = height * buffer_ratio

                top = max( 0.0, top-0.5*height_buffer )
                left = max( 0.0, left-0.5*width_buffer )
                bottom = min( im_height - 1.0, bottom + 0.5*height_buffer )
                right = min( im_width - 1.0, right + 0.5*width_buffer )

                box_infos = []
                for c in box_format:
                    if c == 't':
                        box_infos.append( int(round(top)) ) 
                    elif c == 'l':
                        box_infos.append( int(round(left)) )
                    elif c == 'b':
                        box_infos.append( int(round(bottom)) )
                    elif c == 'r':
                        box_infos.append( int(round(right)) )
                    elif c == 'w':
                        box_infos.append( int(round(width+width_buffer)) )
                    elif c == 'h':
                        box_infos.append( int(round(height+height_buffer)) )
                    else:
                        assert False,'box_format given in detect unrecognised!'
                assert len(box_infos) > 0 ,'box infos is blank'

                detection = (box_infos, cls_conf, cls_name)
                frame_dets.append(detection)
            detections.append(frame_dets)

        return detections

    def _post_processing(self, img, output):
        boxes = []
        for i in range(len(output)):
            boxes.append(self._get_region_boxes(output[i][0], output[i][1]))

        if img.shape[0] > 1:
            bboxs_for_imgs = [
                boxes[0][index] + boxes[1][index] + boxes[2][index]
                for index in range(img.shape[0])]
            # do nms for every detection
            boxes = [self._nms(bboxs) for bboxs in bboxs_for_imgs]
        else:
            boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
            boxes = [self._nms(boxes)]

        return boxes

    def _get_region_boxes(self, boxes, confs):
        # boxes: [batch, num_anchors * H * W, num_classes, 4]
        # confs: [batch, num_anchors * H * W, num_classes]

        # [batch, num_anchors * H * W, num_classes, 4] --> [batch, num_anchors * H * W, 4]
        boxes = boxes[:, :, 0, :]

        all_boxes = []
        for b in range(boxes.shape[0]):
            l_boxes = []

            # [num_anchors * H * W, num_classes] --> [num_anchors * H * W]
            max_conf = confs[b, :, :].max(axis=1)
            # [num_anchors * H * W, num_classes] --> [num_anchors * H * W]
            max_id = confs[b, :, :].argmax(axis=1)

            argwhere = np.argwhere(max_conf > self.thresh)
            max_conf = max_conf[argwhere].flatten()
            max_id = max_id[argwhere].flatten()

            bcx = boxes[b, argwhere, 0]
            bcy = boxes[b, argwhere, 1]
            bw = boxes[b, argwhere, 2]
            bh = boxes[b, argwhere, 3]

            for i in range(bcx.shape[0]):
                l_box = [bcx[i], bcy[i], bw[i], bh[i], max_conf[i], max_conf[i], max_id[i]]
                l_boxes.append(l_box)

            all_boxes.append(l_boxes)
        return all_boxes

    def _nms(self, boxes):
        if len(boxes) == 0:
            return boxes

        det_confs = np.zeros(len(boxes))
        for i in range(len(boxes)):
            det_confs[i] = 1 - boxes[i][4]

        sortIds = np.argsort(det_confs)
        out_boxes = []

        for i in range(len(boxes)):
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
                for j in range(i + 1, len(boxes)):
                    box_j = boxes[sortIds[j]]
                    if self._bbox_iou(box_i, box_j, x1y1x2y2=False) > self.nms_thresh:
                        # print(box_i, box_j, self._bbox_iou(box_i, box_j, x1y1x2y2=False))
                        box_j[4] = 0

        return out_boxes

    @staticmethod
    def _bbox_iou(box1, box2, x1y1x2y2=True):
        if x1y1x2y2:
            mx = min(box1[0], box2[0])
            Mx = max(box1[2], box2[2])
            my = min(box1[1], box2[1])
            My = max(box1[3], box2[3])
            w1 = box1[2] - box1[0]
            h1 = box1[3] - box1[1]
            w2 = box2[2] - box2[0]
            h2 = box2[3] - box2[1]
        else:
            w1 = box1[2]
            h1 = box1[3]
            w2 = box2[2]
            h2 = box2[3]

            mx = min(box1[0], box2[0])
            Mx = max(box1[0] + w1, box2[0] + w2)
            my = min(box1[1], box2[1])
            My = max(box1[1] + h1, box2[1] + h2)
        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh
        carea = 0
        if cw <= 0 or ch <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        return carea / uarea


if __name__ == '__main__':
    import cv2

    imgpath = 'test.jpg'
    vidpath = 'videos/hallway.mp4'
    cfgfile = 'cfg/yolov4.cfg'
    weightfile = 'weights/yolov4.weights'

    device = 0
    yolov4 = YOLOV4( 
      score=0.5, 
      bgr=True, 
      # batch_size=num_vid_streams,
      # gpu_usage=od_gpu_usage, 
      gpu_device='cuda:{}'.format(device),
      # model_image_size=(416, 736)
      model_image_size=(608, 608),
      max_batch_size = 4,
      half=True
    )
    img = cv2.imread(imgpath)
    bs = 5
    imgs = [ img for _ in range(bs) ]
    # img2 = cv2.resize(img, (200,200))

    n = 10
    dur = 0
    for _ in range(n):
        torch.cuda.synchronize()
        tic = perf_counter()
        # yolov4.detect(cfgfile, weightfile, imgpath)
        # yolov4.detect_cv2_camera(cfgfile, weightfile)
        dets = yolov4.detect_get_box_in(imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)[0]
        # print('detections: {}'.format(dets))
        torch.cuda.synchronize()
        toc = perf_counter()
        dur += toc - tic
    print('Average time taken: {:0.3f}s'.format(dur/n))

    cv2.namedWindow('', cv2.WINDOW_NORMAL)
    draw_frame = img.copy()
    for det in dets:
        # print(det)
        bb, score, class_ = det 
        l,t,r,b = bb
        cv2.rectangle(draw_frame, (l,t), (r,b), (255,255,0), 1 )
    
    cv2.imwrite('test_out.jpg', draw_frame)
    cv2.imshow('', draw_frame)
    cv2.waitKey(0)