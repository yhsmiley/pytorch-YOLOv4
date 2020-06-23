import cv2
from time import perf_counter
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class YOLOV4(object):
    if CWD == THIS_DIR:
        _defaults = {
            "engine_path": "trt_weights/yolov4_1_3_608_608_sim.trt",
            "classes_path": 'data/coco.names',
            "thresh": 0.5,
            "nms_thresh": 0.4,
            "model_image_size": (608,608) # must follow trt size
        }
    else:
        _defaults = {
            "engine_path": "pytorch_YOLOv4/trt_weights/yolov4_1_3_608_608_sim.trt",
            "classes_path": 'pytorch_YOLOv4/data/coco.names',
            "thresh": 0.5,
            "nms_thresh": 0.4,
            "model_image_size": (608,608) # must follow trt size
        }

    def __init__(self, bgr=True, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        # for portability between keras-yolo3/yolo.py and this
        if 'model_path' in kwargs:
            kwargs['weights'] = kwargs['model_path']
        if 'score' in kwargs:
            kwargs['thresh'] = kwargs['score']
        self.__dict__.update(kwargs) # update with user overrides

        self.trt_engine = self.get_engine(self.engine_path)
        self.trt_context = self.trt_engine.create_execution_context()
        self.max_batch_size = self.trt_engine.max_batch_size

        self.class_names = self._get_class()
        self.bgr = bgr

        # warm up
        self._detect([np.zeros((10,10,3), dtype=np.uint8)])
        print('Warmed up!')

    def _get_class(self):
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def get_engine(engine_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_path))
        with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(min_severity=trt.Logger.ERROR)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    @staticmethod
    def allocate_buffers(engine, batch_size=1):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    @staticmethod
    def trt_inference(context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def _detect(self, list_of_imgs):
        if self.bgr:
            list_of_imgs = [ cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in list_of_imgs ]

        resized = [np.array(cv2.resize(img, self.model_image_size)) for img in list_of_imgs]
        images = np.stack(resized, axis=0)
        images = np.divide(images, 255, dtype=np.float32)
        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)
        images = np.ascontiguousarray(images)

        batches = []
        for i in range(0, len(images), self.max_batch_size):
            these_imgs = images[i:i+self.max_batch_size]
            batches.append(these_imgs)

        feature_list = []
        for batch in batches:
            bs = len(batch)
            self.trt_buffers = self.allocate_buffers(self.trt_engine, batch_size=bs)
            inputs, outputs, bindings, stream = self.trt_buffers
            inputs[0].host = batch

            trt_outputs = self.trt_inference(self.trt_context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            features = trt_outputs[0].reshape(-1, 22743, 4 + len(self.class_names))
            features = features[:bs]

            feature_list.append(features)

        feature_list = np.concatenate(feature_list, axis=0)

        return self._post_processing(feature_list)

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

    def _post_processing(self, output):
        # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        # num_anchors = 9
        # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # strides = [8, 16, 32]
        # anchor_step = len(anchors) // num_anchors

        # [batch, num, 4]
        box_array = output[:, :, :4]

        # [batch, num, num_classes]
        confs = output[:, :, 4:]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        bboxes_batch = []
        for i in range(box_array.shape[0]):
           
            argwhere = max_conf[i] > self.thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            keep = self._nms_cpu(l_box_array, l_max_conf)
            
            bboxes = []
            if (keep.size > 0):
                l_box_array = l_box_array[keep, :]
                l_max_conf = l_max_conf[keep]
                l_max_id = l_max_id[keep]

                for j in range(l_box_array.shape[0]):
                    bboxes.append([l_box_array[j, 0], l_box_array[j, 1], l_box_array[j, 2], l_box_array[j, 3], l_max_conf[j], l_max_conf[j], l_max_id[j]])
            
            bboxes_batch.append(bboxes)

        return bboxes_batch

    def _nms_cpu(self, boxes, confs, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= self.nms_thresh)[0]
            order = order[inds + 1]
        
        return np.array(keep)


if __name__ == '__main__':
    import cv2

    imgpath = 'test.jpg'
    img = cv2.imread(imgpath)
    bs = 5
    imgs = [ img for _ in range(bs) ]

    yolov4 = YOLOV4( 
      score=0.5, 
      bgr=True
    )

    n = 10
    dur = 0
    for _ in range(n):
        tic = perf_counter()
        dets = yolov4.detect_get_box_in(imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)[0]
        # print('detections: {}'.format(dets))
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