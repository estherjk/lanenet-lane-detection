#!/usr/bin/env python3
"""
test LaneNet model on video
"""
import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_src', type=str, help='The video source')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def test_lanenet(video_src, weights_path):
    """

    :param video_src:
    :param weights_path:
    :return:
    """
    assert ops.exists(video_src), '{:s} not exist'.format(video_src)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        # Read from video source
        # Reference: https://github.com/MaybeShewill-CV/lanenet-lane-detection/issues/383#issuecomment-635898002
        cap = cv2.VideoCapture(video_src)
        while True:
            ret, image = cap.read()
            if not ret:
                break

            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            t_cost = time.time() - t_start
            LOG.info('Single image inference cost time: {:.5f}s'.format(t_cost))
            LOG.info('FPS: {:.5f}s'.format(1 / t_cost))

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )

            cv2.imshow('result', postprocess_result['source_image'])

            # Press 'Q' to quit
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.video_src, args.weights_path)
