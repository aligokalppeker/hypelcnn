import tensorflow as tf
import tensorflow.contrib.slim as slim
from hyperopt import hp

from NNModel import NNModel
from common_nn_operations import ModelOutputTensors


class CNNModelv2(NNModel):

    def create_tensor_graph(self, model_input_params, class_count, algorithm_params):
        with tf.device(model_input_params.device_id):
            with slim.arg_scope([slim.conv2d, slim.fully_connected]):
                level1_filter_count = int(1200/10)
                data_format = 'NHWC'
                # net = tf.transpose(model_input_params.x, [0, 3, 1, 2])  # Convert to NCHW
                net = slim.conv2d(model_input_params.x, level1_filter_count, [1, 1], scope='conv1',
                                  data_format=data_format,
                                  activation_fn=lambda inp: slim.nn.leaky_relu(inp,
                                                                               alpha=algorithm_params["lrelu_alpha"]))
                level2_filter_count = int(level1_filter_count / 2)
                net = slim.conv2d(net, level2_filter_count, [3, 3], scope='conv2',
                                  data_format=data_format,
                                  activation_fn=lambda inp: slim.nn.leaky_relu(inp,
                                                                               alpha=algorithm_params["lrelu_alpha"]))
                level3_filter_count = int(level2_filter_count / 2)
                net = slim.conv2d(net, level3_filter_count, [5, 5], scope='conv3',
                                  data_format=data_format,
                                  activation_fn=lambda inp: slim.nn.leaky_relu(inp,
                                                                               alpha=algorithm_params["lrelu_alpha"]))
                net = slim.flatten(net)

                net = slim.fully_connected(net, class_count * 9, activation_fn=None, scope='fc1')
                net = slim.dropout(net, algorithm_params["drop_out_ratio"], is_training=model_input_params.is_training)
                net = slim.fully_connected(net, class_count * 6, activation_fn=None, scope='fc2')
                net = slim.dropout(net, algorithm_params["drop_out_ratio"], is_training=model_input_params.is_training)
                net = slim.fully_connected(net, class_count * 3, activation_fn=None, scope='fc3')
                net = slim.dropout(net, algorithm_params["drop_out_ratio"], is_training=model_input_params.is_training)
                net = slim.fully_connected(net, class_count, activation_fn=None, scope='fc4')
        return ModelOutputTensors(y_conv=net, image_output=None, image_original=None)

    def get_loss_func(self, tensor_output, label):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,
                                                          logits=tensor_output.y_conv)

    def get_hyper_param_space(self):
        return {
            'drop_out_ratio': hp.uniform('drop_out_ratio', 0.1, 0.5),
            'learning_rate': hp.uniform('learning_rate', 1e-8, 1e-2),
            'lrelu_alpha': hp.uniform('lrelu_alpha', 0.1, 0.2),
            'learning_rate_decay_factor': 0.96,
            'learning_rate_decay_step': 350,
            'batch_size': hp.choice('batch_size', [16, 32, 48, 64, 96])
        }

    def get_default_params(self, batch_size):
        return {
            "drop_out_ratio": 0.3,
            "learning_rate": 1e-4,
            "learning_rate_decay_factor": 0.96,
            "learning_rate_decay_step": 350,
            "lrelu_alpha": 0.2,
            "batch_size": batch_size
        }
