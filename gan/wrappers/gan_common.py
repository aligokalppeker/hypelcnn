from __future__ import division, absolute_import, print_function

import json
import os
import random
from json import JSONDecodeError

import numpy
import tensorflow as tf
import tensorflow_gan as tfgan
from matplotlib import pyplot as plt
from tensorflow import reduce_mean, reduce_sum
from tensorflow.python.ops.math_ops import reduce_std
from tensorflow.python.training import summary_io

from tensorflow.python.summary.summary import scalar
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.learning_rate_decay import polynomial_decay
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.training_util import get_or_create_global_step, get_global_step

model_generator_name = "Generator"
model_base_name = "Model"

input_x_tensor_name = "x"
input_y_tensor_name = "y"


def adj_shadow_ratio(shadow_ratio, is_shadow): return 1. / shadow_ratio if is_shadow else shadow_ratio


class InitializerHook(tf.estimator.SessionRunHook):

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

    def add_point(self, iteration, diver_val):
        iteration = int(iteration)  # For serialization purposes int64 => int
        diver_val = float(diver_val)  # For serialization purposes float64 => float
        insert_idx = 0
        for (curr_iter, curr_diver) in self.data_holder:
            if diver_val > curr_diver:
                insert_idx = insert_idx + 1

        self.data_holder.insert(insert_idx, (iteration, diver_val))
        if len(self.data_holder) > self.max_size:
            self.data_holder.pop()

    def get_best_diver(self):
        return self.data_holder[0][1] if self.data_holder else None

    def get_point_with_itr(self, iteration):
        result = (None, None)
        for (curr_iter, curr_diver) in self.data_holder:
            if curr_iter == iteration:
                result = (curr_iter, curr_diver)
                break

        return result

    def load(self, file_address):
        try:
            with open(file_address, "rb") as read_file:
                self.data_holder = json.load(read_file)
            print(f"Best ratio file {file_address} is loaded.", self.data_holder)
        except IOError:
            print(f"File {file_address} file not found. No best ratio is loaded.")
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


class BaseValidationHook(SessionRunHook):
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
        self.best_mean_div_holder = BestRatioHolder(10)
        self.best_upper_div_holder = BestRatioHolder(10)
        self.validation_itr_mark = False

    def after_create_session(self, session, coord):
        self._global_step_tensor = get_global_step()

    def _is_validation_itr(self, current_iteration):
        result = True
        if self._iteration_frequency != 0:
            result = current_iteration % self._iteration_frequency == 1 and current_iteration != 1
        return result

    def get_best_mean_div(self):
        return self.best_mean_div_holder.get_best_diver()

    def get_best_upper_div(self):
        return self.best_upper_div_holder.get_best_diver()


class PeerValidationHook(SessionRunHook):
    def __init__(self, *validation_base_hooks):
        self._validation_base_hooks = validation_base_hooks

    def after_create_session(self, session, coord):
        for validation_base_hook in self._validation_base_hooks:
            validation_base_hook.after_create_session(session, coord)

    def after_run(self, run_context, run_values):
        ratio_holder_list = []
        for validation_base_hook in self._validation_base_hooks:
            validation_base_hook.after_run(run_context, run_values)
            ratio_holder_list.append(validation_base_hook.best_mean_div_holder)
        if self._validation_base_hooks[0].validation_itr_mark:
            print("Best common options:",
                  BestRatioHolder.create_common_iterations(ratio_holder_list[0], ratio_holder_list[1]))

    def get_best_mean_div(self):
        return [val_base_hook.get_best_mean_div() for val_base_hook in self._validation_base_hooks
                if val_base_hook.get_best_mean_div() is not None]

    def get_best_upper_div(self):
        return [val_base_hook.get_best_upper_div() for val_base_hook in self._validation_base_hooks
                if val_base_hook.get_best_upper_div() is not None]


class ValidationHook(BaseValidationHook):

    def __init__(self, iteration_freq, sample_count, log_dir, loader, data_set, neighborhood, shadow_map, shadow_ratio,
                 input_tensor, infer_model, name_suffix, fetch_shadows):
        super().__init__(iteration_freq, log_dir, shadow_ratio)
        self._writer = None
        self._infer_model = infer_model
        self._input_tensor = input_tensor
        self._name_suffix = name_suffix
        self._plt_name = f"band_ratio_{name_suffix}"
        self._best_mean_div_addr = os.path.join(self._log_dir, f"best_ratio_{name_suffix}.json")
        self.best_mean_div_holder.load(self._best_mean_div_addr)
        self._bands = loader.get_band_measurements()
        self._data_sample_list = load_samples_for_testing(data_set, sample_count, neighborhood,
                                                          shadow_map, fetch_shadows=fetch_shadows)
        self._div_tensor_mean, self._div_tensor_upper, self._ratio_tensor, self._mean_tensor, self._std_tensor = \
            create_stats_tensor(self._infer_model, self._input_tensor, shadow_ratio)
        self._div_mean_summ = scalar(f"divergence_{name_suffix}", self._div_tensor_mean, collections=["custom"])

        for idx, _data_sample in enumerate(self._data_sample_list):
            self._data_sample_list[idx] = numpy.expand_dims(_data_sample, axis=0)
        self._data_sample_list = numpy.squeeze(numpy.asarray(self._data_sample_list), axis=1)

    def after_create_session(self, session, coord):
        self._global_step_tensor = tf.compat.v1.train.get_global_step()
        self._writer = summary_io.SummaryWriterCache.get(self._log_dir)

    def after_run(self, run_context, run_values):
        current_iter = run_context.session.run(
            self._global_step_tensor) if self._global_step_tensor is not None else 0

        self.validation_itr_mark = self._is_validation_itr(current_iter)
        if self.validation_itr_mark:
            ratio, mean, std, div_mean, div_upper, div_mean_summ = run_context.session.run(
                [self._ratio_tensor, self._mean_tensor, self._std_tensor,
                 self._div_tensor_mean, self._div_tensor_upper, self._div_mean_summ],
                feed_dict={self._input_tensor: self._data_sample_list})
            self.best_mean_div_holder.add_point(current_iter, div_mean)
            self.best_mean_div_holder.save(self._best_mean_div_addr)

            self.best_upper_div_holder.add_point(current_iter, div_upper)

            self._writer.add_summary(div_mean_summ, current_iter)
            self.print_stats(current_iter, div_mean, div_upper, mean, ratio, std)

    def print_stats(self, current_iteration, div_mean, div_upper, mean, ratio, std):
        print(f"Validation metrics for {self._name_suffix} #{current_iteration}")
        print_overall_info(mean, std)
        plot_overall_info(self._bands,
                          numpy.percentile(ratio, 50, axis=0),
                          numpy.percentile(ratio, 10, axis=0),
                          numpy.percentile(ratio, 90, axis=0),
                          current_iteration, self._plt_name, self._log_dir)
        print(f"Divergence for {self._name_suffix}; mean:{div_mean}, upper:{div_upper}")
        print(f"Best {self._name_suffix} options:{self.best_mean_div_holder}")


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

    return tf.cond(pred=global_step < lr_constant_steps, true_fn=lambda: base_lr, false_fn=_lr_decay)


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

    tf.compat.v1.summary.scalar("generator_lr", gen_lr)
    tf.compat.v1.summary.scalar("discriminator_lr", dis_lr)
    return train_ops


def create_inference_for_matrix_input(input_tensor, is_shadow_graph, clip_invalid_values, generator_fn):
    first_dim_idx = 1
    second_dim_idx = 2
    shp = input_tensor.get_shape()
    output_tensor_in_col = []
    for first_dim in range(shp[first_dim_idx]):
        output_tensor_in_row = []
        for second_dim in range(shp[second_dim_idx]):
            input_cell = tf.expand_dims(tf.expand_dims(input_tensor[:, first_dim, second_dim], 1), 1)
            generated_tensor = generator_fn(input_cell)
            if clip_invalid_values:
                input_mean = reduce_mean(input_cell)
                generated_mean = reduce_mean(generated_tensor)
                result_tensor = tf.cond(
                    pred=tf.less(generated_mean, input_mean) if is_shadow_graph else tf.greater(generated_mean,
                                                                                                input_mean),
                    true_fn=lambda: generated_tensor,
                    false_fn=lambda: input_cell)
            else:
                result_tensor = generated_tensor
            output_tensor_in_row.append(result_tensor)
        output_tensor_in_col.append(tf.concat(output_tensor_in_row, axis=second_dim_idx))
    return tf.concat(output_tensor_in_col, axis=first_dim_idx)


def create_input_tensor(data_set, is_shadow_graph):
    element_size = data_set.get_data_shape()
    element_size = [None, element_size[0], element_size[1], data_set.get_casi_band_count()]
    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=element_size,
                                            name=input_x_tensor_name if is_shadow_graph else input_y_tensor_name)
    return input_tensor


def create_stats_tensor(generate_y_tensor, images_x_input_tensor, shadow_ratio):
    def kl_divergence(p, q):
        return reduce_sum(tf.compat.v1.where(tf.not_equal(p, 0.), p * tf.math.log(p / q), tf.zeros_like(p)))

    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    ratio_tensor = tf.squeeze(generate_y_tensor / images_x_input_tensor * shadow_ratio, axis=[1, 2])
    finite_map_tensor = tf.reduce_all(input_tensor=tf.math.is_finite(ratio_tensor), axis=1)
    ratio_tensor_inf_eliminated = ratio_tensor[finite_map_tensor]
    mean = reduce_mean(ratio_tensor_inf_eliminated, axis=0)
    std = reduce_std(ratio_tensor_inf_eliminated, axis=0)
    div_mean = tf.abs(js_divergence(tf.abs(mean - 1), tf.zeros_like(mean)))
    div_upper = tf.abs(js_divergence(tf.abs(mean + std - 1), tf.zeros_like(mean)))
    return div_mean, div_upper, ratio_tensor_inf_eliminated, mean, std


def calculate_stats_from_samples(sess, data_sample_list, images_x_input_tensor, generate_y_tensor, shadow_ratio,
                                 log_dir, current_iteration, plt_name, bands):
    def divergence_for_ratios(mean_val):
        def kl_divergence(p, q):
            return numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))

        def js_divergence(p, q):
            m = 0.5 * (p + q)
            return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

        return abs(js_divergence(numpy.abs(mean_val - 1), numpy.zeros_like(mean_val)))

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


def load_samples_for_testing(data_set, sample_count, neighborhood, shadow_map, fetch_shadows):
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
        data_sample_list.append(
            data_set.get_data_point(sample_indice_in_image[0], sample_indice_in_image[1])[:, :, 0:band_size])
    return data_sample_list


def read_hsi_data(loader, data_set, shadow_map, pairing_method, sampling_method_map):
    if pairing_method not in sampling_method_map:
        raise ValueError(f"Wrong sampling parameter value ({pairing_method}).")
    normal_data_as_matrix, shadow_data_as_matrix = \
        sampling_method_map[pairing_method].get_sample_pairs(data_set, loader, shadow_map)
    normal_data_as_matrix = normal_data_as_matrix[:, :, :, 0:data_set.get_casi_band_count()]
    shadow_data_as_matrix = shadow_data_as_matrix[:, :, :, 0:data_set.get_casi_band_count()]
    return normal_data_as_matrix, shadow_data_as_matrix


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
        if band_index == 0:
            prefix = "[ "
        elif band_index == band_size - 1:
            postfix = " ]"

        print(f"{prefix}{mean[band_index]:2.4f}\u00B1{std[band_index]:2.2f}{postfix}",
              end="\n" if band_index % 5 == 1 else " ")
