import argparse
import cv2
import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time

from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

from pi_camera import video

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt_runtime = trt.Runtime(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file', type=str, help="ONNX file (.onnx)")
    parser.add_argument('--video_src', type=str, help="Video source. For Pi Camera, use 'pi_camera'.")
    parser.add_argument('--engine_file', type=str, help="TensorRT engine file (.engine)")

    return parser.parse_args()


def build_engine(onnx_file, engine_file):
    """
    Takes an ONNX file and creates a TensorRT engine.
    References:
      - https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/
      - https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics
      - https://www.learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
      - https://github.com/NVIDIA/TensorRT/issues/319#issuecomment-571918258
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        if builder.platform_has_fast_fp16:
            builder.fp16_mode = True

        if not os.path.exists(onnx_file):
            print('ONNX file {} not found.'.format(onnx_file))
            exit(0)

        print('Loading ONNX file from path {}...'.format(onnx_file))
        with open(onnx_file, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

            print('Beginning ONNX file parsing...')
            parser.parse(model.read())

        print('Parsing ONNX file complete.')

        print('Building an engine from file {}; this may take a while...'.format(onnx_file))
        engine = builder.build_cuda_engine(network)
        print('Building engine complete.')

        save_engine(engine, engine_file)

        return engine


def save_engine(engine, filename):
    """
    Save TensorRT engine.
    NOTE: Serialized engines are not portable across platforms or TensorRT versions.
    """
    serialized_engine = engine.serialize()
    with open(filename, 'wb') as f:
        f.write(serialized_engine)


def load_engine(filename):
    """
    Load TensorRT engine.
    """
    with open(filename, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    return engine


def do_inference(engine, video_src):
    """
    Use TensorRT to run inference.
    """
    
    # Create LaneNet postprocessor
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
    
    # Allocate memory for inputs & outputs

    input_shape = engine.get_binding_shape(0)
    input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
    device_input = cuda.mem_alloc(input_size)

    output_binary_shape = engine.get_binding_shape(1)
    host_output_binary = cuda.pagelocked_empty(trt.volume(output_binary_shape) * engine.max_batch_size, dtype=np.float32)
    device_output_binary = cuda.mem_alloc(host_output_binary.nbytes)

    output_pixel_embedding_shape = engine.get_binding_shape(2)
    host_output_pixel_embedding = cuda.pagelocked_empty(trt.volume(output_pixel_embedding_shape) * engine.max_batch_size, dtype=np.float32)
    device_output_pixel_embedding = cuda.mem_alloc(host_output_pixel_embedding.nbytes)

    stream = cuda.Stream()

    with engine.create_execution_context() as context:
        # Read from video source
        if video_src == 'pi_camera':
            cap = cv2.VideoCapture(video.gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(video_src)

        while True:
            ret, image = cap.read()
            if not ret:
                break

            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            # Transfer input data to GPU
            host_input = np.array(image, dtype=np.float32, order='C')
            cuda.memcpy_htod_async(device_input, host_input, stream)

            # Run inference
            t_start = time.time()
            # context.profiler = trt.Profiler()
            context.execute(batch_size=1, bindings=[
                int(device_input), 
                int(device_output_binary),
                int(device_output_pixel_embedding)
            ])

            # Transfer predictions back from the GPU
            cuda.memcpy_dtoh_async(host_output_binary, device_output_binary, stream)
            cuda.memcpy_dtoh_async(host_output_pixel_embedding, device_output_pixel_embedding, stream)
            
            # Synchronize the stream
            stream.synchronize()
            t_cost = time.time() - t_start
            LOG.info('Single image inference cost time: {:.5f}s'.format(t_cost))
            LOG.info('FPS: {:.5f}s'.format(1 / t_cost))

            # Postprocess result
            postprocess_result = postprocessor.postprocess(
                binary_seg_result=host_output_binary.reshape(output_binary_shape),
                instance_seg_result=host_output_pixel_embedding.reshape(output_pixel_embedding_shape),
                source_image=image_vis
            )

            # Display result
            if postprocess_result['source_image'] is not None:
                cv2.imshow('result', postprocess_result['source_image'])

                t_cost = time.time() - t_start
                LOG.info('Inference + postprocess + display cost time: {:.5f}s'.format(t_cost))
                LOG.info('FPS: {:.5f}s'.format(1 / t_cost))

            # Press 'Q' to quit
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    args = init_args()

    if os.path.exists(args.engine_file):
        print("Loading TensorRT engine...")
        engine = load_engine(args.engine_file)
    else:
        print("Building TensorRT engine...")
        engine = build_engine(args.onnx_file, args.engine_file)

    do_inference(engine, args.video_src)