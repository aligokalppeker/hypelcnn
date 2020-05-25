import math

import tensorflow as tf
from hyperopt import hp
from tensorflow import initializers
from tensorflow.contrib import slim as slim

from NNModel import NNModel
from common_nn_operations import ModelOutputTensors


# hyperopt old result :
# {
#   "batch_size": 48,
#   "drop_out_ratio": 0.3055297115008223,
#   "learning_rate": 0.0002937010046830672,
#   "learning_rate_decay_factor": 0.96,
#   "learning_rate_decay_step": 350,
#   "lrelu_alpha": 0.1802203883485628,
#   "filter_count": 4800
# }
class CNNModelv4(NNModel):

    def get_hyper_param_space(self):
        return {
            'drop_out_ratio': hp.uniform('drop_out_ratio', 0.3, 0.5),
            'learning_rate': hp.uniform('learning_rate', 1e-5, 1e-3),
            'lrelu_alpha': 0.18,
            'learning_rate_decay_factor': 0.96,
            'learning_rate_decay_step': 350,
            'filter_count': 1200,
            'batch_size': hp.choice('batch_size', [32, 48, 64]),
            'optimizer': 'AdamOptimizer',
            'bn_decay': hp.uniform('bn_decay', 0.900, 0.999),
            'l2regularizer_scale': 0.00001,
            'spectral_hierarchy_level': 3,
            'spatial_hierarchy_level': 3,
            'degradation_coeff': 3
        }

    # 'optimizer': hp.choice('optimizer',['AdamOptimizer', ('MomentumOptimizer', hp.uniform('momentum', 0.50, 0.99))])

    def get_default_params(self, batch_size):
        return {
            "drop_out_ratio": 0.3,
            "learning_rate": 1e-4,
            "learning_rate_decay_factor": 0.96,
            "learning_rate_decay_step": 350,
            "lrelu_alpha": 0.2,
            "filter_count": 1200,
            "batch_size": batch_size,
            "optimizer": "AdamOptimizer",
            "bn_decay": 0.95,
            "l2regularizer_scale": 0.00001,
            "spectral_hierarchy_level": 3,
            "spatial_hierarchy_level": 3,
            "degradation_coeff": 3
        }

    def create_tensor_graph(self, model_input_params, class_count, algorithm_params):
        with tf.device(model_input_params.device_id):
            data_format = None  # 'NHWC'
            bn_training_params = {'is_training': model_input_params.is_training, 'decay': algorithm_params["bn_decay"]}
            lrelu = lambda inp: slim.nn.leaky_relu(inp, alpha=algorithm_params["lrelu_alpha"])
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=initializers.variance_scaling(scale=2.0),
                                weights_regularizer=slim.l2_regularizer(algorithm_params["l2regularizer_scale"]),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=bn_training_params,
                                activation_fn=lrelu):
                level_filter_count = algorithm_params["filter_count"]

                if data_format == 'NCHW':
                    net0 = tf.transpose(model_input_params.x, [0, 3, 1, 2])  # Convert input to NCHW
                else:
                    net0 = model_input_params.x

                spectral_hierarchy_level = algorithm_params["spectral_hierarchy_level"]
                net1 = self.__create_spectral_nn_layers(data_format, level_filter_count, net0, spectral_hierarchy_level,
                                                        True)
                net1 = net1 + CNNModelv4.__scale_input_to_output(net0, net1)

                net2 = self.__create_spectral_nn_layers(data_format, level_filter_count, net1, spectral_hierarchy_level,
                                                        False)
                net2 = net2 + CNNModelv4.__scale_input_to_output(net1, net2)

                spatial_hierarchy_level = algorithm_params["spatial_hierarchy_level"]
                net3 = self.__create_levels_as_blocks(data_format,
                                                      int(net2.get_shape()[3].value / 2),
                                                      net2, spatial_hierarchy_level)
                net3 = net3 + CNNModelv4.__scale_input_to_output(net2, net3)

                net4 = slim.flatten(net3)

                degradation_coeff = algorithm_params["degradation_coeff"]
                net5 = self.__create_fc_block(algorithm_params, class_count, degradation_coeff, model_input_params,
                                              net4)

                net6 = slim.fully_connected(net5, class_count, weights_regularizer=None, activation_fn=None,
                                            scope='fc_final')

                image_gen_net4 = None
                if model_input_params.is_training:
                    image_gen_net1 = slim.fully_connected(net6, class_count * 3, weights_regularizer=None,
                                                          scope='image_gen_net_1')
                    image_gen_net2 = slim.fully_connected(image_gen_net1, class_count * 9, weights_regularizer=None,
                                                          scope='image_gen_net_2')
                    image_gen_net3 = slim.fully_connected(image_gen_net2, class_count * 27, weights_regularizer=None,
                                                          scope='image_gen_net_3')
                    image_size = (net0.get_shape()[1] * net0.get_shape()[2] * net0.get_shape()[3]).value
                    image_gen_net4 = slim.fully_connected(image_gen_net3, image_size, weights_regularizer=None,
                                                          activation_fn=tf.sigmoid,
                                                          scope='image_gen_net_4')
        return ModelOutputTensors(y_conv=net6, image_output=image_gen_net4, image_original=net0)

    def get_loss_func(self, tensor_output, label):
        original_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=tensor_output.y_conv)
        if tensor_output.image_output is None:
            total_loss = original_loss
        else:
            original_reshaped = tf.reshape(tensor_output.image_original,
                                           [-1, tensor_output.image_output.get_shape()[1]])
            reconstruction_err = tf.reduce_mean(tf.square(tensor_output.image_output - original_reshaped))
            total_loss = original_loss + reconstruction_err
            tf.losses.add_loss(total_loss)

        return total_loss

    @staticmethod
    def __create_fc_block(algorithm_params, class_count, degradation_coeff, model_input_params, netinput):
        flatten_element_size = netinput.get_shape()[1].value
        fc_stage_count = math.floor(math.log(flatten_element_size / class_count, degradation_coeff))
        element_size = flatten_element_size
        for drop_out_stage_index in range(0, fc_stage_count - 1):
            element_size = int(element_size / degradation_coeff)
            netinput = slim.fully_connected(netinput, element_size, weights_regularizer=None,
                                            scope='fc_' + str(drop_out_stage_index))
            netinput = slim.dropout(netinput, keep_prob=1 - algorithm_params["drop_out_ratio"],
                                    is_training=model_input_params.is_training)
        return netinput

    @staticmethod
    def __create_levels_as_blocks(data_format, level_final_filter_count, netinput, block_level_count):
        for index in range(0, block_level_count):
            next_net = CNNModelv4.__create_a_level(int(level_final_filter_count / pow(2, index)), netinput,
                                                   'connector_' + str(index),
                                                   data_format)
            # disabling the residual connection
            # compatibility issues with older network models(before 27.04.20)
            # next_net = next_net + CNNModelv4.__scale_input_to_output(netinput, next_net)

            next_net_conv = slim.conv2d(next_net, next_net.get_shape()[3], [1, 1],
                                        scope='connector_conv_' + str(index),
                                        data_format=data_format)
            netinput = next_net_conv + next_net
        return netinput

    @staticmethod
    def __create_spectral_nn_layers(data_format, level_final_filter_count, net0, count, is_encoding_layers):
        net_input = net0
        for nn_index in range(0, count):
            if is_encoding_layers:
                layer_filter_size = level_final_filter_count / pow(2, ((count - 1) - nn_index))
                conv_name = 'conv_enc_'
            else:
                layer_filter_size = level_final_filter_count / pow(2, nn_index)
                conv_name = 'conv_dec_'

            next_net = slim.conv2d(net_input, int(layer_filter_size), [1, 1],
                                   scope=conv_name + str(nn_index),
                                   data_format=data_format)
            next_net = next_net + CNNModelv4.__scale_input_to_output(net_input, next_net)
            net_input = next_net
        return net_input

    @staticmethod
    def __create_a_level(level_filter_count, input_data, level_name, data_format):
        patch_size = input_data.get_shape()[1].value
        with tf.name_scope(level_name + '_scope'):
            elements = []
            for patch_x_index in range(1, patch_size + 1):
                for patch_y_index in range(1, patch_size + 1):
                    # compatibility issues with older network models(before 27.04.20)
                    if patch_x_index % 2 == 1 and patch_y_index % 2 == 1 and patch_x_index == patch_y_index:
                        scope_name = level_name + '_conv' + str(patch_x_index) + 'x' + str(patch_y_index)
                        level_element = slim.conv2d(input_data, level_filter_count, [patch_x_index, patch_y_index],
                                                    scope=scope_name,
                                                    data_format=data_format)
                        elements.append(level_element)

            net = tf.concat(axis=3, values=elements)
        return net

    @staticmethod
    def __scale_input_to_output(input_data, output_data):
        axis_no = 3

        input_channel_size = input_data.get_shape()[axis_no].value
        output_channel_size = output_data.get_shape()[axis_no].value
        scale_ratio = input_channel_size / output_channel_size

        output_data_indice_list = []
        for output_data_index in range(0, output_channel_size):
            target_index_no = min(round(output_data_index * scale_ratio), input_channel_size - 1)
            output_data_indice_list.append(target_index_no)

        return tf.gather(input_data, output_data_indice_list, axis=axis_no)

    @staticmethod
    # This method is kept for compatibility issues with older network models(before 27.04.20)
    def __scale_input_to_output_legacy(input_data, output_data):
        axis_no = 3

        input_channel_size = input_data.get_shape()[axis_no].value
        output_channel_size = output_data.get_shape()[axis_no].value
        scale_ratio = input_channel_size / output_channel_size

        input_data_list = tf.unstack(input_data, axis=axis_no)
        output_data_list = []
        for output_data_index in range(0, output_channel_size):
            target_index_no = min(round(output_data_index * scale_ratio), input_channel_size - 1)
            output_data_list.append(input_data_list[target_index_no])

        return tf.stack(output_data_list, axis=axis_no)
