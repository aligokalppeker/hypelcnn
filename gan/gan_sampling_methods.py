from __future__ import division, absolute_import, print_function

from abc import abstractmethod, ABC

import numpy
from scipy import ndimage

from DataLoader import SampleSet


class Sampler(ABC):
    @abstractmethod
    def get_sample_pairs(self, data_set, loader, shadow_map):
        pass


class NeighborhoodBasedSampler(Sampler):

    def __init__(self, neighborhood_size, margin) -> None:
        self._margin = margin
        self._neighborhood_size = neighborhood_size

    def get_sample_pairs(self, data_set, loader, shadow_map):
        data_shape_info = data_set.get_data_shape()

        shadow_element_count = numpy.sum(shadow_map, dtype=numpy.int)
        shadow_data_as_matrix = numpy.zeros(numpy.concatenate([[shadow_element_count], data_shape_info]),
                                            dtype=numpy.float32)

        non_shadow_map = ndimage.binary_dilation(shadow_map, iterations=self._neighborhood_size).astype(
            shadow_map.dtype) - ndimage.binary_dilation(shadow_map, iterations=self._margin).astype(shadow_map.dtype)

        normal_element_count = numpy.sum(non_shadow_map, dtype=numpy.int)
        normal_data_as_matrix = numpy.zeros(numpy.concatenate([[normal_element_count], data_shape_info]),
                                            dtype=numpy.float32)

        shadow_element_index = 0
        normal_element_index = 0
        for x_index in range(0, shadow_map.shape[0]):
            for y_index in range(0, shadow_map.shape[1]):
                point_value = loader.get_point_value(data_set, [y_index, x_index])
                if shadow_map[x_index, y_index] == 1:
                    shadow_data_as_matrix[shadow_element_index, :, :, :] = point_value
                    shadow_element_index = shadow_element_index + 1
                elif non_shadow_map[x_index, y_index] == 1:
                    normal_data_as_matrix[normal_element_index, :, :, :] = point_value
                    normal_element_index = normal_element_index + 1

        # Trim normal data to shadow data count, we assume that there is more normal data
        normal_data_as_matrix = normal_data_as_matrix[0:shadow_data_as_matrix.shape[0], :, :, :]
        return normal_data_as_matrix, shadow_data_as_matrix


class RandomBasedSampler(Sampler):

    def __init__(self, multiply_shadowed_data) -> None:
        self._multiply_shadowed_data = multiply_shadowed_data

    def get_sample_pairs(self, data_set, loader, shadow_map):
        data_shape_info = data_set.get_data_shape()
        shadow_element_count = numpy.sum(shadow_map, dtype=numpy.int)
        normal_element_count = shadow_map.shape[0] * shadow_map.shape[1] - shadow_element_count
        shadow_data_as_matrix = numpy.zeros(numpy.concatenate([[shadow_element_count], data_shape_info]),
                                            dtype=numpy.float32)
        normal_data_as_matrix = numpy.zeros(numpy.concatenate([[normal_element_count], data_shape_info]),
                                            dtype=numpy.float32)

        shadow_element_index = 0
        normal_element_index = 0
        for x_index in range(0, shadow_map.shape[0]):
            for y_index in range(0, shadow_map.shape[1]):
                point_value = loader.get_point_value(data_set, [y_index, x_index])
                if shadow_map[x_index, y_index] == 1:
                    shadow_data_as_matrix[shadow_element_index, :, :, :] = point_value
                    shadow_element_index = shadow_element_index + 1
                else:
                    normal_data_as_matrix[normal_element_index, :, :, :] = point_value
                    normal_element_index = normal_element_index + 1

        # Duplicate shadow data to match size of normal data
        if self._multiply_shadowed_data:
            shadow_data_as_matrix = numpy.repeat(shadow_data_as_matrix, repeats=(
                    normal_element_count // shadow_element_count), axis=0)

        # Trim normal data to shadow data count, we assume that there is more normal data
        normal_data_as_matrix = normal_data_as_matrix[0:shadow_data_as_matrix.shape[0], :, :, :]

        return normal_data_as_matrix, shadow_data_as_matrix


class TargetBasedSampler(Sampler):
    def __init__(self, margin):
        self._margin = margin

    def get_sample_pairs(self, data_set, loader, shadow_map):
        samples = SampleSet(training_targets=loader.read_targets("shadow_cycle_gan/class_result.tif"),
                            test_targets=None,
                            validation_targets=None)
        first_margin_start = self._margin
        first_margin_end = data_set.get_scene_shape()[0] - self._margin
        second_margin_start = self._margin
        second_margin_end = data_set.get_scene_shape()[1] - self._margin
        for target_index in range(0, samples.training_targets.shape[0]):
            current_target = samples.training_targets[target_index]
            if not (first_margin_start < current_target[1] < first_margin_end and
                    second_margin_start < current_target[0] < second_margin_end):
                current_target[2] = -1
        normal_data_as_matrix, shadow_data_as_matrix = self._get_targetbased_shadowed_normal_data(data_set,
                                                                                                  loader,
                                                                                                  shadow_map,
                                                                                                  samples)
        return normal_data_as_matrix, shadow_data_as_matrix

    @staticmethod
    def _get_targetbased_shadowed_normal_data(data_set, loader, shadow_map, samples):
        if samples.test_targets is None and samples.training_targets is not None:
            all_targets = samples.training_targets
        elif samples.training_targets is None and samples.test_targets is not None:
            all_targets = samples.test_targets
        else:
            all_targets = numpy.vstack([samples.test_targets, samples.training_targets])

        class_count = loader.get_class_count().stop
        # First Pass for target size map creation
        shadow_target_count_map = numpy.zeros([class_count, 1], numpy.int32)
        normal_target_count_map = numpy.zeros([class_count, 1], numpy.int32)
        for target_index in range(0, all_targets.shape[0]):
            current_target = all_targets[target_index]
            target_id = current_target[2]
            if target_id >= 0:
                if shadow_map[current_target[1], current_target[0]] == 1:
                    shadow_target_count_map[target_id] = shadow_target_count_map[target_id] + 1
                else:
                    normal_target_count_map[target_id] = normal_target_count_map[target_id] + 1

        # Second Pass for target based array creation
        shadow_target_map = {}
        normal_target_map = {}
        data_shape = data_set.get_data_shape()
        for target_key in range(0, class_count):
            shadow_target_count = shadow_target_count_map[target_key]
            if shadow_target_count > 0:
                shadow_target_map[target_key] = numpy.empty(numpy.concatenate([shadow_target_count, data_shape]),
                                                            numpy.float32)
            normal_target_count = normal_target_count_map[target_key]
            if normal_target_count > 0:
                normal_target_map[target_key] = numpy.empty(numpy.concatenate([normal_target_count, data_shape]),
                                                            numpy.float32)

        # Third Pass for data assignment
        shadow_target_counter_map = numpy.zeros([class_count, 1], numpy.int32)
        normal_target_counter_map = numpy.zeros([class_count, 1], numpy.int32)
        for target_index in range(0, all_targets.shape[0]):
            current_target = all_targets[target_index]
            point_value = loader.get_point_value(data_set, current_target)
            target_id = current_target[2]
            if target_id >= 0:
                if shadow_map[current_target[1], current_target[0]] == 1:
                    shadow_target_map[target_id][shadow_target_counter_map[target_id]] = point_value
                    shadow_target_counter_map[target_id] = shadow_target_counter_map[target_id] + 1
                else:
                    normal_target_map[target_id][normal_target_counter_map[target_id]] = point_value
                    normal_target_counter_map[target_id] = normal_target_counter_map[target_id] + 1

        normal_data_as_matrix = None
        shadow_data_as_matrix = None
        for target_key, shadow_point_values in shadow_target_map.items():
            if target_key in normal_target_map:
                normal_point_values = normal_target_map[target_key]
                normal_point_count = normal_point_values.shape[0]
                shadow_point_count = shadow_point_values.shape[0]
                shadow_data_multiplier = int(normal_point_count / shadow_point_count)
                shadow_data_reminder = normal_point_count % shadow_point_count
                shadow_point_expanded_values = numpy.repeat(shadow_point_values, repeats=shadow_data_multiplier, axis=0)
                shadow_point_expanded_values = numpy.vstack(
                    [shadow_point_expanded_values, shadow_point_expanded_values[0:shadow_data_reminder, :, :, :]])
                if normal_data_as_matrix is None:
                    normal_data_as_matrix = normal_point_values
                else:
                    normal_data_as_matrix = numpy.vstack([normal_data_as_matrix, normal_point_values])

                if shadow_data_as_matrix is None:
                    shadow_data_as_matrix = shadow_point_expanded_values
                else:
                    shadow_data_as_matrix = numpy.vstack([shadow_data_as_matrix, shadow_point_expanded_values])
            else:
                print("Shadow target key is not found in normal target list:", target_key)
        return normal_data_as_matrix, shadow_data_as_matrix


class DummySampler(Sampler):
    def __init__(self, element_count, fill_value, coefficient):
        self._element_count = element_count
        self._fill_value = fill_value
        self._coefficient = coefficient

    def get_sample_pairs(self, data_set, loader, shadow_map):
        data_shape_info = data_set.get_data_shape()
        shadow_data_as_matrix = numpy.full(numpy.concatenate([[self._element_count], data_shape_info]),
                                           fill_value=self._fill_value, dtype=numpy.float32)
        return shadow_data_as_matrix * self._coefficient, shadow_data_as_matrix
