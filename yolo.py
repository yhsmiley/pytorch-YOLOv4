import cv2
from time import perf_counter
import numpy as np
import torch

import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()
if CWD == THIS_DIR:
	# from tool.darknet2pytorch import Darknet
	from models import Yolov4
else:
	# from pytorch_YOLOv4.tool.darknet2pytorch import Darknet
	from pytorch_YOLOv4.models import Yolov4


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
		# self._detect([np.zeros((10,10,3), dtype=np.uint8)])
		self._detect([np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8)])
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
		if self.bgr:
			list_of_imgs = [ cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in list_of_imgs ]

		resized = [np.array(cv2.resize(img, self.model_image_size)) for img in list_of_imgs]
		images = np.stack(resized, axis=0)
		images = np.divide(images, 255, dtype=np.float32)
		images = torch.from_numpy(images.transpose(0, 3, 1, 2))

		images = images.cuda(self.device)
		images = torch.autograd.Variable(images)

		if self.half:
			images = images.half()

		batches = []
		for i in range(0, len(images), self.max_batch_size):
			these_imgs = images[i:i+self.max_batch_size]
			batches.append(these_imgs)

		feature_list = None
		with torch.no_grad():
			for batch in batches:
				features = self.model(batch)

				if feature_list is None:
					feature_list = features
				else:
					feature_list = torch.cat((feature_list, features))

		output = feature_list.cpu().numpy()

		return self._post_processing(output)


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
	vidpath = 'videos/hallway.mp4'
	cfgfile = 'cfg/yolov4.cfg'
	weightfile = 'weights/yolov4.weights'

	device = 0
	# Height in {320, 416, 512, 608, ... 320 + 96 * n}
	# Width in  {320, 416, 512, 608, ... 320 + 96 * m}
	yolov4 = YOLOV4( 
	  score=0.5, 
	  bgr=True, 
	  # batch_size=num_vid_streams,
	  # gpu_usage=od_gpu_usage, 
	  gpu_device='cuda:{}'.format(device),
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