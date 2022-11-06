import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import leaky_relu
from tf_slim import conv2d, dropout, flatten, fully_connected, arg_scope

from nnmodel.NNModel import NNModel
from common.common_nn_ops import ModelOutputTensors


class DUALCNNModel(NNModel):

    # TODO: Move to hyper param json files
    # def get_hyper_param_space(self, trial):
    #     return {
    #         "drop_out_ratio": trial.suggest_float("drop_out_ratio", 0.1, 0.5),
    #         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3),
    #         "lrelu_alpha": trial.suggest_float("lrelu_alpha", 0.1, 0.2),
    #         "learning_rate_decay_factor": 0.96,
    #         "learning_rate_decay_step": 350,
    #         "filter_count": trial.suggest_categorical("filter_count", [100, 200, 400, 800]),
    #         "batch_size": trial.suggest_categorical("batch_size", [32, 48, 64, 96]),
    #         "optimizer": "AdamOptimizer"
    #     }

    def create_tensor_graph(self, model_input_params, class_count, algorithm_params):
        with tf.device(model_input_params.device_id):
            with arg_scope([conv2d, fully_connected],
                           # normalizer_fn=batch_norm,
                           # normalizer_params={"is_training": model_input_params.is_training, "decay": 0.95},
                           # weights_initializer=initializers.variance_scaling(scale=2.0),
                           # weights_regularizer=l2_regularizer(algorithm_params["l2regularizer_scale"]),
                           activation_fn=lambda inp: leaky_relu(inp, alpha=algorithm_params["lrelu_alpha"])):
                band_size = model_input_params.x.get_shape()[3].value
                hs_lidar_groups = tf.split(axis=3, num_or_size_splits=[band_size - 1, 1],
                                           value=model_input_params.x)

                hs_lidar_diff = algorithm_params["hs_lidar_diff"]
                hs_net = self._create_hs_tensor_branch(algorithm_params,
                                                       hs_lidar_groups[0][:, hs_lidar_diff:-hs_lidar_diff,
                                                       hs_lidar_diff:-hs_lidar_diff, :])
                lidar_net = self._create_lidar_tensor_branch(hs_lidar_groups[1])

                net = tf.concat(axis=1, values=[flatten(hs_net), flatten(lidar_net)])
                net = self._create_fc_tensor_branch(algorithm_params, class_count, model_input_params, net)
        return ModelOutputTensors(y_conv=net, image_output=None, image_original=None, histogram_tensors=[])

    @staticmethod
    def _create_lidar_tensor_branch(lidar_group):
        net = DUALCNNModel._create_a_level(2, lidar_group, "lidar_level1")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="lidar_connector_conv1")
        net = DUALCNNModel._create_a_level(4, net, "lidar_level2")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="lidar_connector_conv2")
        net = DUALCNNModel._create_a_level(8, net, "lidar_level3")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="lidar_connector_conv3")
        return net

    @staticmethod
    def _create_fc_tensor_branch(algorithm_params, class_count, model_input_params, net):
        net = flatten(net)
        net = fully_connected(net, class_count * 9, scope="fc1")
        net = dropout(net, algorithm_params["drop_out_ratio"], is_training=model_input_params.is_training)
        net = fully_connected(net, class_count * 6, scope="fc2")
        net = dropout(net, algorithm_params["drop_out_ratio"], is_training=model_input_params.is_training)
        net = fully_connected(net, class_count * 3, scope="fc3")
        net = dropout(net, algorithm_params["drop_out_ratio"], is_training=model_input_params.is_training)
        net = fully_connected(net, class_count, activation_fn=None, scope="fc4")
        return net

    @staticmethod
    def _create_hs_tensor_branch(algorithm_params, hs_group):
        level_filter_count = algorithm_params["filter_count"]

        net = DUALCNNModel._create_a_level(level_filter_count // 4, hs_group, "level1")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="connector_conv1")

        net = DUALCNNModel._create_a_level(level_filter_count // 2, net, "level2")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="connector_conv2")

        net = DUALCNNModel._create_a_level(level_filter_count, net, "level3")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="connector_conv3")

        net = DUALCNNModel._create_a_level(level_filter_count // 2, net, "level4")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="connector_conv4")

        net = DUALCNNModel._create_a_level(level_filter_count // 4, net, "level5")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="connector_conv5")

        net = DUALCNNModel._create_a_level(level_filter_count // 8, net, "level6")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="connector_conv6")

        net = DUALCNNModel._create_a_level(level_filter_count // 16, net, "level7")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="connector_conv7")

        net = DUALCNNModel._create_a_level(level_filter_count // 32, net, "level8")
        net = conv2d(net, net.get_shape()[3], [1, 1], scope="connector_conv8")

        return net

    def get_loss_func(self, tensor_output, label):
        return tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                       logits=tensor_output.y_conv)

    @staticmethod
    def _create_a_level(level_filter_count, input_data, level_name):
        patch_size = input_data.get_shape()[1].value
        with tf.compat.v1.name_scope(level_name + "_scope"):
            elements = []
            for patch_index in range(1, patch_size + 1):
                if patch_index % 2 == 1:
                    scope_name = level_name + "_conv" + str(patch_index) + "x" + str(patch_index)
                    level_element = conv2d(input_data, level_filter_count, [patch_index, patch_index],
                                           scope=scope_name)
                    elements.append(level_element)

            net = tf.concat(axis=3, values=elements)
        return net
