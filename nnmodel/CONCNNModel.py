import tensorflow as tf
from tf_slim import conv2d, dropout, flatten, fully_connected, arg_scope

from common.common_nn_ops import ModelOutputTensors
from nnmodel.NNModel import NNModel


class CONCNNModel(NNModel):

    # TODO: Move to hyper param json files
    # def get_hyper_param_space(self, trial):
    #     return {
    #         "filter_count": 128,
    #         "drop_out_ratio": trial.suggest_float("drop_out_ratio", 0.1, 0.5),
    #         "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1e-2),
    #         "lrelu_alpha": trial.suggest_float("lrelu_alpha", 0.1, 0.2),
    #         "learning_rate_decay_factor": 0.96,
    #         "learning_rate_decay_step": 350,
    #         "batch_size": trial.suggest_categorical("batch_size", [16, 32, 48, 64, 96]),
    #         "optimizer": "AdamOptimizer"
    #     }

    def create_tensor_graph(self, model_input_params, class_count, algorithm_params):
        with tf.device(model_input_params.device_id):
            with arg_scope([conv2d, fully_connected]):
                level0_filter_count = algorithm_params["filter_count"]
                data_format = None
                if data_format == 'NCHW':
                    net0 = tf.transpose(a=model_input_params.x, perm=[0, 3, 1, 2])  # Convert input to NCHW
                else:  # "NHWC"
                    net0 = model_input_params.x

                net0_1x1 = conv2d(net0, level0_filter_count, [1, 1], scope="conv0_1x1", data_format=data_format)
                net0_3x3 = conv2d(net0, level0_filter_count, [3, 3], scope="conv0_3x3", data_format=data_format)
                net0_5x5 = conv2d(net0, level0_filter_count, [5, 5], scope="conv0_5x5", data_format=data_format)
                net0_out = tf.concat(axis=3, values=[net0_1x1, net0_3x3, net0_5x5])
                net0_out = tf.nn.local_response_normalization(net0_out)

                level1_filter_count = level0_filter_count * 3
                net11 = conv2d(net0_out, level1_filter_count, [1, 1], scope="conv11", data_format=data_format)
                net11 = tf.nn.local_response_normalization(net11)
                net12 = conv2d(net11, level1_filter_count, [1, 1], scope="conv12", data_format=data_format)
                net13 = conv2d(net12, level1_filter_count, [1, 1], scope="conv13", data_format=data_format)
                net13 = net13 + net11

                level2_filter_count = level1_filter_count
                net21 = conv2d(net13, level2_filter_count, [1, 1], scope="conv21", data_format=data_format)
                net22 = conv2d(net21, level2_filter_count, [1, 1], scope="conv22", data_format=data_format)
                net22 = net22 + net13

                level3_filter_count = level2_filter_count
                net31 = conv2d(net22, level3_filter_count, [1, 1], scope="conv31", data_format=data_format)
                net31 = dropout(net31, algorithm_params["drop_out_ratio"],
                                is_training=model_input_params.is_training)

                net32 = conv2d(net31, level3_filter_count, [1, 1], scope="conv32", data_format=data_format)
                net32 = dropout(net32, algorithm_params["drop_out_ratio"],
                                is_training=model_input_params.is_training)

                net33 = conv2d(net32, level3_filter_count, [1, 1], scope="conv33", data_format=data_format)

                net_33 = flatten(net33)
                net_fc = fully_connected(net_33, class_count, activation_fn=None, scope="fc")
        return ModelOutputTensors(y_conv=net_fc, image_output=None, image_original=None, histogram_tensors=[])

    def get_loss_func(self, tensor_output, label):
        return tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                       logits=tensor_output.y_conv)
