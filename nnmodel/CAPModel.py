import numpy
import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import leaky_relu
from tf_slim import conv2d, fully_connected, arg_scope, batch_norm

from nnmodel.NNModel import NNModel
from common.common_nn_ops import ModelOutputTensors


class CAPModel(NNModel):
    # TODO: Move to hyper param json files
    # def get_hyper_param_space(self, trial):
    #     return {
    #         "iter_routing": trial.suggest_categorical("iter_routing", [3, 4, 5]),
    #         "conv_layer_kernel_size": 3,
    #         "primary_caps_kernel_size": 3,
    #         "feature_count": trial.suggest_categorical("feature_count", [256, 378, 512]),
    #         "primary_capsule_count": trial.suggest_categorical("primary_capsule_count", [64, 80, 96]),
    #         "primary_capsule_output_space": 8,
    #         "digit_capsule_output_space": 16,
    #         "lrelu_alpha": trial.suggest_float("lrelu_alpha", 0.1, 0.2),
    #         "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1e-2),
    #         "learning_rate_decay_factor": 0.96,
    #         "learning_rate_decay_step": 350,
    #         "batch_size": trial.suggest_categorical("batch_size", [16, 32, 48, 64, 96]),
    #         "enable_decoding": trial.suggest_categorical("enable_decoding", [True, False]),
    #         "optimizer": "AdamOptimizer"
    #     }

    def create_tensor_graph(self, model_input_params, class_count, algorithm_params):
        iter_routing = algorithm_params["iter_routing"]
        conv_layer_kernel_size = [algorithm_params["conv_layer_kernel_size"],
                                  algorithm_params["conv_layer_kernel_size"]]
        primary_caps_kernel_size = [algorithm_params["primary_caps_kernel_size"],
                                    algorithm_params["primary_caps_kernel_size"]]
        feature_count = algorithm_params["feature_count"]
        primary_capsule_count = algorithm_params["primary_capsule_count"]
        primary_capsule_output_space = algorithm_params["digit_capsule_output_space"]
        digit_capsule_output_space = algorithm_params["digit_capsule_output_space"]

        digit_capsule_count = class_count
        batch_size = -1
        enable_decoding = algorithm_params["enable_decoding"]

        lrelu_func = lambda inp: leaky_relu(inp, alpha=algorithm_params["lrelu_alpha"])
        with tf.device(model_input_params.device_id):
            with arg_scope([conv2d], trainable=model_input_params.is_training
                           # weights_initializer=initializers.variance_scaling(scale=2.0),
                           # activation_fn=lrelu_func
                           # weights_regularizer=slim.l2_regularizer(0.00001),
                           # normalizer_fn=slim.batch_norm,
                           ):
                with tf.compat.v1.variable_scope("Conv1_layer") as scope:
                    image_output = conv2d(model_input_params.x,
                                          num_outputs=feature_count,
                                          kernel_size=conv_layer_kernel_size,
                                          padding="VALID",
                                          scope=scope, normalizer_fn=batch_norm)

                with tf.compat.v1.variable_scope("PrimaryCaps_layer") as scope:
                    image_output = conv2d(image_output,
                                          num_outputs=primary_capsule_count * primary_capsule_output_space,
                                          kernel_size=primary_caps_kernel_size, stride=1,
                                          padding="VALID",
                                          scope=scope, normalizer_fn=batch_norm)
                    data_size = (image_output.get_shape()[1] *
                                 image_output.get_shape()[2] *
                                 image_output.get_shape()[3]).value // primary_capsule_output_space
                    image_output = tf.reshape(image_output, [batch_size, data_size, 1, primary_capsule_output_space])

                with tf.compat.v1.variable_scope("DigitCaps_layer"):
                    u_hats = []
                    image_output_groups = tf.split(axis=1, num_or_size_splits=data_size, value=image_output)
                    for i in range(data_size):
                        u_hat = conv2d(image_output_groups[i],
                                       num_outputs=digit_capsule_count * digit_capsule_output_space,
                                       kernel_size=[1, 1],
                                       padding="VALID",
                                       scope="DigitCaps_layer_w_" + str(i), activation_fn=None)
                        u_hat = tf.reshape(u_hat,
                                           [batch_size, 1, digit_capsule_count, digit_capsule_output_space])
                        u_hats.append(u_hat)

                    image_output = tf.concat(u_hats, axis=1)

                    b_ijs = tf.constant(numpy.zeros([data_size, digit_capsule_count], dtype=numpy.float32))
                    v_js = []
                    for r_iter in range(iter_routing):
                        with tf.compat.v1.variable_scope("iter_" + str(r_iter)):
                            b_ij_groups = tf.split(axis=1, num_or_size_splits=digit_capsule_count, value=b_ijs)

                            c_ijs = tf.nn.softmax(b_ijs, axis=1)
                            c_ij_groups = tf.split(axis=1, num_or_size_splits=digit_capsule_count, value=c_ijs)

                            image_output_groups = tf.split(axis=2, num_or_size_splits=digit_capsule_count,
                                                           value=image_output)
                            for i in range(digit_capsule_count):
                                c_ij = tf.reshape(tf.tile(c_ij_groups[i], [1, digit_capsule_output_space]),
                                                  [c_ij_groups[i].get_shape()[0], 1, digit_capsule_output_space, 1])
                                s_j = tf.nn.depthwise_conv2d(input=image_output_groups[i], filter=c_ij,
                                                             strides=[1, 1, 1, 1],
                                                             padding="VALID")
                                # Squash function
                                s_j = tf.reshape(s_j, [batch_size, digit_capsule_output_space])
                                s_j_norm_square = tf.reduce_mean(input_tensor=tf.square(s_j), axis=1, keepdims=True)
                                v_j = s_j_norm_square * s_j / ((1 + s_j_norm_square) * tf.sqrt(s_j_norm_square + 1e-9))

                                b_ij_groups[i] = b_ij_groups[i] + tf.reduce_sum(
                                    input_tensor=tf.matmul(tf.reshape(image_output_groups[i],
                                                                      [batch_size,
                                                                       image_output_groups[i].get_shape()[1],
                                                                       digit_capsule_output_space]),
                                                           tf.reshape(v_j,
                                                                      [batch_size, digit_capsule_output_space, 1])),
                                    axis=0)

                                if r_iter == iter_routing - 1:
                                    v_js.append(tf.reshape(v_j, [batch_size, 1, digit_capsule_output_space]))

                            b_ijs = tf.concat(b_ij_groups, axis=1)

                    image_output = tf.concat(v_js, axis=1)

                    with tf.compat.v1.variable_scope("Masking"):
                        y_conv = tf.norm(tensor=image_output, axis=2)

                    decoder_image_output = None
                    if model_input_params.is_training and enable_decoding:
                        y_as_float = tf.cast(model_input_params.y, dtype=tf.float32)
                        masked_v = tf.matmul(image_output,
                                             tf.reshape(y_as_float, [batch_size, digit_capsule_count, 1]),
                                             transpose_a=True)
                        masked_v = tf.reshape(masked_v, [batch_size, digit_capsule_output_space])

                        with tf.compat.v1.variable_scope("Decoder"):
                            size = (model_input_params.x.get_shape()[1] *
                                    model_input_params.x.get_shape()[2] *
                                    model_input_params.x.get_shape()[3]).value
                            image_output = fully_connected(masked_v, 512, scope="fc1",
                                                           activation_fn=lrelu_func,
                                                           trainable=model_input_params.is_training)
                            image_output = fully_connected(image_output, 1024, scope="fc2",
                                                           activation_fn=lrelu_func,
                                                           trainable=model_input_params.is_training)
                            decoder_image_output = fully_connected(image_output, size, scope="fc3",
                                                                   activation_fn=tf.sigmoid,
                                                                   trainable=model_input_params.is_training)

        return ModelOutputTensors(y_conv=y_conv, image_output=decoder_image_output, image_original=model_input_params.x,
                                  histogram_tensors=[])

    @staticmethod
    def _capsule_loss(logits, labels, x_output, x_original):
        labels_as_float = tf.cast(labels, dtype=tf.float32)

        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5
        max_l = tf.square(tf.maximum(0., m_plus - logits))
        max_r = tf.square(tf.maximum(0., logits - m_minus))

        l_c = labels_as_float * max_l + lambda_val * (1 - labels_as_float) * max_r

        margin_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=l_c, axis=1))

        if x_output is None:
            total_loss = margin_loss
        else:
            origin = tf.reshape(x_original, [-1, x_output.get_shape()[1]])
            reconstruction_err = tf.reduce_mean(input_tensor=tf.square(x_output - origin))
            total_loss = margin_loss + 0.0005 * reconstruction_err

        tf.compat.v1.losses.add_loss(total_loss)

        return total_loss

    def get_loss_func(self, tensor_output, label):
        original_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=tensor_output.y_conv)
        if tensor_output.image_output is None:
            total_loss = original_loss
        else:
            original_reshaped = tf.reshape(tensor_output.image_original,
                                           [-1, tensor_output.image_output.get_shape()[1]])
            reconstruction_err = tf.reduce_mean(input_tensor=tf.square(tensor_output.image_output - original_reshaped))
            total_loss = original_loss + reconstruction_err
            tf.compat.v1.losses.add_loss(total_loss)

        return total_loss

    def __get_loss_func(self, tensor_output, label):
        return self._capsule_loss(labels=label, logits=tensor_output.y_conv,
                                  x_output=tensor_output.image_output,
                                  x_original=tensor_output.image_original)
