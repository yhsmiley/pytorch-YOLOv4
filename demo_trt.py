import sys
import os
import time
import argparse
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from time import perf_counter

from tool.utils import *

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def GiB(val):
    return val * 1 << 30

def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.
    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.
    Returns:
        str: Path of data directory.
    Raises:
        FileNotFoundError
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory.", default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " + data_root + " instead.")
        data_path = data_root

    # Make sure data directory exists.
    if not (os.path.exists(data_path)):
        raise FileNotFoundError(data_path + " does not exist. Please provide the correct data path with the -d option.")

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(find_files[index] + " does not exist. Please provide the correct data path with the -d option.")

    return data_path, find_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
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
def do_inference(context, bindings, inputs, outputs, stream):
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

TRT_LOGGER = trt.Logger()

def main(engine_path, image_path, image_size, img_bs):
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        image_src = cv2.imread(image_path)
        image_src_batch = [ image_src for _ in range(img_bs) ]

        print('engine max batch size: {}'.format(engine.max_batch_size))

        num_classes = 80

        n = 10
        dur = 0
        for i in range(n):
            tic = perf_counter()
            boxes = detect(engine, context, image_src_batch, image_size, num_classes)
            # print('boxes: {}'.format(boxes))
            toc = perf_counter()
            if i != 0:
                dur += toc - tic
        print('Average time taken: {:0.3f}s'.format(dur/(n-1)))

        if num_classes == 20:
            namesfile = 'data/voc.names'
        elif num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'

        class_names = load_class_names(namesfile)
        plot_boxes_cv2(image_src, boxes[0], savename='predictions_trt.jpg', class_names=class_names)

def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def detect(engine, context, image_src_batch, image_size, num_classes):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    ta = time.time()
    # Input
    image_src_batch = [ cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_src_batch ]
    resized = [np.array(cv2.resize(img, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)) for img in image_src_batch]
    img_in = np.stack(resized, axis=0)
    img_in = np.divide(img_in, 255, dtype=np.float32)
    img_in = np.transpose(img_in, (0, 3, 1, 2)).astype(np.float32)
    img_in = np.ascontiguousarray(img_in)

    print("Shape of the input image batch: ", img_in.shape)
    # print(img_in)

    batches = []
    for i in range(0, len(img_in), engine.max_batch_size):
        these_imgs = img_in[i:i+engine.max_batch_size]
        batches.append(these_imgs)

    trt_output_list = []
    for batch in batches:
        trt_buffers = allocate_buffers(engine)
        inputs, outputs, bindings, stream = trt_buffers
        inputs[0].host = batch

        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print('Len of outputs: ', len(trt_outputs))

        # (19*19 + 38*38 + 76*76) * 3 = 22743 for 608x608
        trt_output = trt_outputs[0].reshape(-1, 22743, 4 + num_classes)
        print('trt output shape: {}'.format(trt_output.shape))
        trt_output = trt_output[:len(batch)]
        trt_output_list.append(trt_output)

    trt_output_list = np.concatenate(trt_output_list, axis=0)

    tb = time.time()

    print('trt shape: {}'.format(trt_output_list.shape))

    print('-----------------------------------')
    print('    TRT inference time: %f' % (tb - ta))
    print('-----------------------------------')

    boxes = post_processing(img_in, 0.5, 0.4, trt_output_list)

    return boxes


if __name__ == '__main__':
    engine_path = sys.argv[1]
    image_path = sys.argv[2]
    
    if len(sys.argv) < 4:
        image_size = (416, 416)
        img_bs = 5
    elif len(sys.argv) < 5:
        image_size = (int(sys.argv[3]), int(sys.argv[3]))
        img_bs = int(sys.argv[4])
    else:
        image_size = (int(sys.argv[3]), int(sys.argv[4]))
        img_bs = int(sys.argv[5])
    
    main(engine_path, image_path, image_size, img_bs)
