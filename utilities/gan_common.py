from __future__ import division, absolute_import, print_function

import json
import os
from json import JSONDecodeError

import numpy
import tensorflow as tf

from shadow_data_generator import load_samples_for_testing, calculate_stats_from_samples


def create_dummy_shadowed_normal_data(data_set, loader):
    data_shape_info = loader.get_data_shape(data_set)
    element_count = 2000
    shadow_data_as_matrix = numpy.full(numpy.concatenate([[element_count], data_shape_info]),
                                       fill_value=0.5, dtype=numpy.float32)

    return shadow_data_as_matrix * 2, shadow_data_as_matrix


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
        self._shadow_ratio = shadow_ratio[0:-1]
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
                 input_tensor, model, name_suffix, fetch_shadows):
        super().__init__(iteration_freq, log_dir, shadow_ratio)
        self._forward_model = model
        self._input_tensor = input_tensor
        self._name_suffix = name_suffix
        self._plt_name = f"band_ratio_{name_suffix}"
        self._best_ratio_addr = os.path.join(self._log_dir, f"best_ratio_{name_suffix}.json")
        self.best_ratio_holder.load(self._best_ratio_addr)
        self._data_sample_list = load_samples_for_testing(loader, data_set, sample_count, neighborhood,
                                                          shadow_map, fetch_shadows=fetch_shadows)
        for idx, _data_sample in enumerate(self._data_sample_list):
            self._data_sample_list[idx] = numpy.expand_dims(_data_sample, axis=0)

    def after_run(self, run_context, run_values):
        session = run_context.session
        if self._global_step_tensor is None:
            current_iteration = 0
        else:
            current_iteration = session.run(self._global_step_tensor)

        self.validation_itr_mark = self._is_validation_itr(current_iteration)
        if self.validation_itr_mark:
            print(f"Validation metrics for {self._name_suffix} #{current_iteration}")
            kl_shadowed = calculate_stats_from_samples(session, self._data_sample_list, self._input_tensor,
                                                       self._forward_model,
                                                       self._shadow_ratio, self._log_dir, current_iteration,
                                                       plt_name=self._plt_name)
            self.best_ratio_holder.add_point(current_iteration, kl_shadowed)
            self.best_ratio_holder.save(self._best_ratio_addr)
            print(f"KL divergence for {self._name_suffix}:{kl_shadowed}")
            print(f"Best {self._name_suffix} options:{self.best_ratio_holder}")
