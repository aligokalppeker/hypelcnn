from __future__ import division, absolute_import, print_function

import json
import os
import random
from json import JSONDecodeError

import numpy
import tensorflow as tf
import tensorflow_gan as tfgan
from matplotlib import pyplot as plt
from tensorflow_core.python import reduce_mean, reduce_std, reduce_sum
from tensorflow_core.python.training.adam import AdamOptimizer
from tensorflow_core.python.training.learning_rate_decay import polynomial_decay
from tensorflow_core.python.training.training_util import get_or_create_global_step

from DataLoader import ShadowOperationStruct

model_generator_name = "Generator"
model_base_name = "Model"

input_x_tensor_name = "x"
input_y_tensor_name = "y"


def adj_shadow_ratio(shadow_ratio, is_shadow):
    return 1. / shadow_ratio if is_shadow else shadow_ratio


class InitializerHook(tf.train.SessionRunHook):

    def __init__(self, input_itr, normal_placeholder, shadow_placeholder, normal_data, shadow_data):
        self.input_itr = input_itr
        self.shadow_data = shadow_data
        self.normal_data = normal_data
        self.shadow_placeholder = shadow_placeholder
        self.normal_placeholder = normal_placeholder

    def after_create_session(self, session, coord):
        session.run(self.input_itr.initializer,
                    feed_dict={self.shadow_placeholder: self.shadow_data,
                               self.normal_placeholder: self.normal_data})


class BestRatioHolder:

    def __init__(self, max_size) -> None:
        super().__init__()
        self.data_holder = []
        self.max_size = max_size

    def add_point(self, iteration, kl_val):
        iteration = int(iteration)  # For seralization purposes int64 => int
        kl_val = float(kl_val)  # For seralization purposes float64 => float
        insert_idx = 0
        for (curr_iter, curr_kl) in self.data_holder:
            if kl_val > curr_kl:
                insert_idx = insert_idx + 1

        self.data_holder.insert(insert_idx, (iteration, kl_val))
        if len(self.data_holder) > self.max_size:
            self.data_holder.pop()

    def get_point_with_itr(self, iteration):
        result = (None, None)
        for (curr_iter, curr_kl) in self.data_holder:
            if curr_iter == iteration:
                result = (curr_iter, curr_kl)
                break

        return result

    def load(self, file_address):
        try:
            with open(file_address, "rb") as read_file:
                self.data_holder = json.load(read_file)
            print(f"Best ratio file {file_address} is loaded.", self.data_holder)
        except IOError:
            print(f"File {file_address} file found. No best ratio is loaded.")
        except JSONDecodeError:
            print(f"File {file_address} file can not be decoded. No best ratio is loaded.")

    def save(self, file_address):
        serialized_out = json.dumps(self.data_holder)
        with open(file_address, "w") as write_file:
            write_file.write(serialized_out)

    @staticmethod
    def create_common_iterations(ratio_holder_1, ratio_holder_2):
        result = BestRatioHolder(ratio_holder_1.max_size)
        for (curr_iter, curr_kl) in ratio_holder_1.data_holder:
            (found_itr, found_kl) = ratio_holder_2.get_point_with_itr(curr_iter)
            if found_itr is not None:
                result.add_point(found_itr, found_kl)

        return result

    def __str__(self) -> str:
        return str(self.data_holder)


class BaseValidationHook(tf.train.SessionRunHook):
    @staticmethod
    def export(sess, input_pl, input_np, output_tensor):
        # Grab a single image and run it through inference
        output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
        return output_np

    def __init__(self, iteration_freq, log_dir, shadow_ratio):
        self._iteration_frequency = iteration_freq
        self._global_step_tensor = None
        self._shadow_ratio = shadow_ratio
        self._log_dir = log_dir
        self.best_ratio_holder = BestRatioHolder(10)
        self.validation_itr_mark = False

    def after_create_session(self, session, coord):
        self._global_step_tensor = tf.train.get_global_step()

    def _is_validation_itr(self, current_iteration):
        result = True
        if self._iteration_frequency != 0:
            result = current_iteration % self._iteration_frequency == 1 and current_iteration != 1
        return result


class PeerValidationHook(tf.train.SessionRunHook):
    def __init__(self, *validation_base_hooks):
        self._validation_base_hooks = validation_base_hooks

    def after_create_session(self, session, coord):
        for validation_base_hook in self._validation_base_hooks:
            validation_base_hook.after_create_session(session, coord)

    def after_run(self, run_context, run_values):
        ratio_holder_list = []
        for validation_base_hook in self._validation_base_hooks:
            validation_base_hook.after_run(run_context, run_values)
            ratio_holder_list.append(validation_base_hook.best_ratio_holder)
        if self._validation_base_hooks[0].validation_itr_mark:
            print("Best common options:",
                  BestRatioHolder.create_common_iterations(ratio_holder_list[0], ratio_holder_list[1]))


class ValidationHook(BaseValidationHook):

    def __init__(self, iteration_freq, sample_count, log_dir, loader, data_set, neighborhood, shadow_map, shadow_ratio,
                 input_tensor, infer_model, name_suffix, fetch_shadows):
        super().__init__(iteration_freq, log_dir, shadow_ratio)
        self._infer_model = infer_model
        self._input_tensor = input_tensor
        self._name_suffix = name_suffix
        self._plt_name = f"band_ratio_{name_suffix}"
        self._best_ratio_addr = os.path.join(self._log_dir, f"best_ratio_{name_suffix}.json")
        self.best_ratio_holder.load(self._best_ratio_addr)
        self._bands = loader.get_band_measurements()
        self._data_sample_list = load_samples_for_testing(loader, data_set, sample_count, neighborhood,
                                                          shadow_map, fetch_shadows=fetch_shadows)
        self._diver_tensor, self._ratio_tensor, self._mean_tensor, self._std_tensor = create_stats_tensor(
            self._infer_model,
            self._input_tensor,
            shadow_ratio)
        for idx, _data_sample in enumerate(self._data_sample_list):
            self._data_sample_list[idx] = numpy.expand_dims(_data_sample, axis=0)
        self._data_sample_list = numpy.squeeze(numpy.asarray(self._data_sample_list), axis=1)

    def after_run(self, run_context, run_values):
        session = run_context.session
        if self._global_step_tensor is None:
            current_iteration = 0
        else:
            current_iteration = session.run(self._global_step_tensor)

        self.validation_itr_mark = self._is_validation_itr(current_iteration)
        if self.validation_itr_mark:
            ratio, mean, std, divergence = session.run(
                [self._ratio_tensor, self._mean_tensor, self._std_tensor, self._diver_tensor],
                feed_dict={self._input_tensor: self._data_sample_list})
            self.best_ratio_holder.add_point(current_iteration, divergence)
            self.best_ratio_holder.save(self._best_ratio_addr)

            self.print_stats(current_iteration, divergence, mean, ratio, std)

    def print_stats(self, current_iteration, divergence, mean, ratio, std):
        print(f"Validation metrics for {self._name_suffix} #{current_iteration}")
        print_overall_info(mean, std)
        plot_overall_info(self._bands,
                          numpy.percentile(ratio, 50, axis=0),
                          numpy.percentile(ratio, 10, axis=0),
                          numpy.percentile(ratio, 90, axis=0),
                          current_iteration, self._plt_name, self._log_dir)
        print(f"Divergence for {self._name_suffix}:{divergence}")
        print(f"Best {self._name_suffix} options:{self.best_ratio_holder}")


def construct_gan_inference_graph(input_data, gan_inference_wrapper):
    with tf.device('/cpu:0'):
        hs_converted = gan_inference_wrapper.construct_inference_graph(tf.expand_dims(input_data[:, :, :-1], axis=0),
                                                                       is_shadow_graph=True,
                                                                       clip_invalid_values=False)
    axis_id = 2
    return tf.concat(axis=axis_id, values=[hs_converted, tf.expand_dims(input_data[:, :, -1], axis=-1)])


def construct_gan_inference_graph_randomized(input_data, wrapper):
    # coin = tf.less(tf.random_uniform([1], 0, 1.0)[0], 0.5)
    # images = tf.cond(coin,
    #                  lambda: GRSS2013DataLoader.construct_cyclegan_inference_graph(input_data,
    #                                                                                model_forward_generator_name),
    #                  lambda: GRSS2013DataLoader.construct_cyclegan_inference_graph(input_data,
    #                                                                                model_backward_generator_name))
    images = construct_gan_inference_graph(input_data, wrapper)
    return images


def construct_simple_shadow_inference_graph(input_data, shadow_ratio):
    # coin = tf.less(tf.random_uniform([1], 0, 1.0)[0], 0.5)
    # images = tf.cond(coin, lambda: input_data / shadow_ratio, lambda: input_data * shadow_ratio)
    images = input_data / shadow_ratio
    return images


def _get_lr(base_lr, max_number_of_steps):
    """Returns a learning rate `Tensor`.

    Args:
      base_lr: A scalar float `Tensor` or a Python number.  The base learning
          rate.

    Returns:
      A scalar float `Tensor` of learning rate which equals `base_lr` when the
      global training step is less than FLAGS.max_number_of_steps / 2, afterwards
      it linearly decays to zero.
    """
    global_step = get_or_create_global_step()
    lr_constant_steps = max_number_of_steps // 2

    def _lr_decay():
        return polynomial_decay(
            learning_rate=base_lr,
            global_step=(global_step - lr_constant_steps),
            decay_steps=(max_number_of_steps - lr_constant_steps),
            end_learning_rate=0.0)

    return tf.cond(global_step < lr_constant_steps, lambda: base_lr, _lr_decay)


def define_standard_train_ops(gan_model, gan_loss, max_number_of_steps,
                              generator_lr, discriminator_lr):
    """Defines train ops that trains `gan_model` with `gan_loss`.

    Args:
      discriminator_lr: Discriminator learning rate
      generator_lr: Generator learning rate
      max_number_of_steps: Number of max steps for learning
      gan_model: A `GANModel` namedtuple.
      gan_loss: A `GANLoss` namedtuple containing all losses for
          `gan_model`.

    Returns:
      A `GANTrainOps` namedtuple.
    """
    gen_lr = _get_lr(generator_lr, max_number_of_steps)
    dis_lr = _get_lr(discriminator_lr, max_number_of_steps)
    gen_opt = AdamOptimizer(gen_lr, beta1=0.5, use_locking=True)
    dis_opt = AdamOptimizer(dis_lr, beta1=0.5, use_locking=True)

    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=gen_opt,
        discriminator_optimizer=dis_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True,
        check_for_unused_update_ops=False,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    tf.summary.scalar("generator_lr", gen_lr)
    tf.summary.scalar("discriminator_lr", dis_lr)
    return train_ops


def create_simple_shadow_struct(shadow_ratio):
    simple_shadow_func = lambda inp: (construct_simple_shadow_inference_graph(inp, shadow_ratio))
    simple_shadow_struct = ShadowOperationStruct(shadow_op=simple_shadow_func, shadow_op_creater=lambda: None,
                                                 shadow_op_initializer=lambda restorer, session: None)
    return simple_shadow_struct


def create_gan_struct(gan_inference_wrapper, model_base_dir, ckpt_relative_path):
    gan_shadow_func = lambda inp: (construct_gan_inference_graph_randomized(inp, gan_inference_wrapper))
    gan_shadow_op_creater = gan_inference_wrapper.create_generator_restorer
    gan_shadow_op_initializer = lambda restorer, session: (
        restorer.restore(session, model_base_dir + ckpt_relative_path))
    gan_struct = ShadowOperationStruct(shadow_op=gan_shadow_func, shadow_op_creater=gan_shadow_op_creater,
                                       shadow_op_initializer=gan_shadow_op_initializer)
    return gan_struct


def create_stats_tensor(generate_y_tensor, images_x_input_tensor, shadow_ratio):
    def kl_divergence(p, q):
        return reduce_sum(tf.where(tf.not_equal(p, 0.), p * tf.log(p / q), tf.zeros_like(p)))

    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    ratio_tensor = tf.squeeze(generate_y_tensor / images_x_input_tensor * shadow_ratio, axis=[1, 2])
    finite_map_tensor = tf.reduce_all(tf.is_finite(ratio_tensor), axis=1)
    ratio_tensor_inf_eliminated = ratio_tensor[finite_map_tensor]
    mean = reduce_mean(ratio_tensor_inf_eliminated, axis=0)
    std = reduce_std(ratio_tensor_inf_eliminated, axis=0)
    divergence = tf.abs(js_divergence(tf.abs(mean - 1), tf.zeros_like(mean)))
    return divergence, ratio_tensor_inf_eliminated, mean, std


def calculate_stats_from_samples(sess, data_sample_list, images_x_input_tensor, generate_y_tensor, shadow_ratio,
                                 log_dir, current_iteration, plt_name, bands):
    def divergence_for_ratios(mean):
        def kl_divergence(p, q):
            return numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))

        def js_divergence(p, q):
            m = 0.5 * (p + q)
            return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

        return abs(js_divergence(numpy.abs(mean - 1), numpy.zeros_like(mean)))

    generated_y_data = sess.run(generate_y_tensor, feed_dict={images_x_input_tensor: data_sample_list})
    ratio = generated_y_data / data_sample_list
    ratio_inf_eliminated = ratio[numpy.isfinite(ratio).all(axis=3)]

    final_ratio = ratio_inf_eliminated * shadow_ratio
    mean = numpy.mean(final_ratio, axis=0)
    std = numpy.std(final_ratio, axis=0)
    divergence = divergence_for_ratios(mean)
    print_overall_info(mean, std)
    plot_overall_info(bands,
                      numpy.percentile(final_ratio, 50, axis=0),
                      numpy.percentile(final_ratio, 10, axis=0),
                      numpy.percentile(final_ratio, 90, axis=0),
                      current_iteration, plt_name, log_dir)
    return divergence


def load_samples_for_testing(loader, data_set, sample_count, neighborhood, shadow_map, fetch_shadows):
    # neighborhood aware indices finder
    band_size = data_set.get_casi_band_count()
    data_sample_list = []
    shadow_check_val = 0

    if neighborhood > 0:
        shadow_map = shadow_map[neighborhood:-neighborhood, neighborhood:-neighborhood]

    if fetch_shadows:
        indices = numpy.where(shadow_map > shadow_check_val)
    else:
        indices = numpy.where(shadow_map == shadow_check_val)

    for index in range(0, sample_count):
        # Pick a random point
        random_indice_in_sample_list = random.randint(0, indices[0].size - 1)
        sample_indice_in_image = [indices[1][random_indice_in_sample_list], indices[0][random_indice_in_sample_list]]
        data_sample_list.append(loader.get_point_value(data_set, sample_indice_in_image)[:, :, 0:band_size])
    return data_sample_list


def plot_overall_info(bands, mean, lower_bound, upper_bound, iteration, plt_name, log_dir):
    # band_idxs = linspace(1, bands.shape[0], bands.shape[0], dtype=numpy.int)
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = 14
    plt.scatter(bands, mean, label="mean ratio", s=10)
    plt.plot(bands, mean)
    plt.fill_between(bands, lower_bound, upper_bound, alpha=0.2)
    # plt.legend(loc='upper left')
    # plt.title("Band ratio")
    plt.xlabel("Spectral band(nm)")
    plt.ylabel("Ratio between generated and original samples")
    plt.ylim([-1, 4])
    plt.yticks(list(range(-1, 5)))
    # plt.yticks(list(range(numpy.round(numpy.min(lower_bound)).astype(int),
    #                       numpy.round(numpy.max(upper_bound)).astype(int) + 1)))
    plt.grid()
    plt.savefig(os.path.join(log_dir, f"{plt_name}_{iteration}.pdf"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


def print_overall_info(mean, std):
    print("Mean&std Generated vs Original Ratio: ")
    band_size = mean.shape[0]
    for band_index in range(0, band_size):
        prefix = ""
        postfix = ""
        end = "  "
        if band_index == 0:
            prefix = "[ "
        elif band_index == band_size - 1:
            postfix = " ]"

        if band_index % 5 == 1:
            end = '\n'
        print("%s%2.4f\u00B1%2.2f%s" % (prefix, mean[band_index], std[band_index], postfix), end=end)
