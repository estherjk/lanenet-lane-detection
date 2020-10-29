"""
Freeze graph. Use TensorBoard to visualize the graph.

To visualize the graph with TensorBoard:
tensorboard --logdir __tb

References:
  - mnn_project/freeze_lanenet_model.py
  - https://github.com/VasinPA/lanenet-lane-detection/blob/master/tools/test_lanenet_and_freeze.py
"""

import argparse
import os
import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt

from lanenet_model import lanenet
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--save_path', type=str, help='File path to write .pb file')

    return parser.parse_args()

def freeze_graph(weights_path, save_path):
    # construct compute graph
    with tf.variable_scope('lanenet'):
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    with tf.variable_scope('lanenet/'):
        binary_seg_ret = tf.cast(binary_seg_ret, dtype=tf.float32)
        binary_seg_ret = tf.squeeze(binary_seg_ret, axis=0, name='final_binary_output')
        instance_seg_ret = tf.squeeze(instance_seg_ret, axis=0, name='final_pixel_embedding_output')

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # create a session
    saver = tf.train.Saver(variables_to_restore)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        # Use TensorBoard to visualize graph
        writer = tf.summary.FileWriter('__tb', sess.graph)

        converted_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def=sess.graph.as_graph_def(),
            output_node_names=[
                'lanenet/input_tensor',
                'lanenet/final_binary_output',
                'lanenet/final_pixel_embedding_output'
            ]
        )

        with tf.gfile.GFile(save_path, "wb") as f:
            f.write(converted_graph_def.SerializeToString())


if __name__ == '__main__':
    args = init_args()

    freeze_graph(
        weights_path=args.weights_path,
        save_path=args.save_path
    )