from collections import namedtuple

import numpy
import scipy.io
from numba import jit
from sklearn.model_selection import StratifiedShuffleSplit
from tifffile import imread

from DataLoader import DataLoader, SampleSet, ShadowOperationStruct
from shadow_data_generator import create_generator_restorer, construct_cyclegan_inference_graph_randomized, \
    construct_simple_shadow_inference_graph

DataSet = namedtuple('DataSet', ['concrete_data', 'shadow_creator_dict', 'neighborhood', 'casi_min', 'casi_max'])


class GRSS2013DataLoader(DataLoader):

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def get_original_data_type(self):
        return numpy.uint16

    def load_data(self, neighborhood, normalize):
        casi = imread(self.get_model_base_dir() + '2013_IEEE_GRSS_DF_Contest_CASI.tif')
        lidar = imread(self.get_model_base_dir() + '2013_IEEE_GRSS_DF_Contest_LiDAR.tif')[:, :, numpy.newaxis]

        # Padding part
        pad_size = ((neighborhood, neighborhood), (neighborhood, neighborhood), (0, 0))
        lidar = numpy.pad(lidar, pad_size, mode='symmetric')
        casi = numpy.pad(casi, pad_size, mode='symmetric')

        casi_min = None
        casi_max = None
        if normalize:
            lidar -= numpy.min(lidar)
            lidar = lidar / numpy.max(lidar)
            casi_min = numpy.min(casi, axis=(0, 1))
            casi -= casi_min
            casi_max = numpy.max(casi, axis=(0, 1))
            casi = casi / casi_max.astype(numpy.float32)

        concrete_data = numpy.zeros([casi.shape[0], casi.shape[1], casi.shape[2] + 1], dtype=numpy.float32)
        concrete_data[:, :, 0:concrete_data.shape[2] - 1] = casi
        concrete_data[:, :, concrete_data.shape[2] - 1] = lidar[:, :, 0]

        data_set_for_shadowing = DataSet(shadow_creator_dict=None, concrete_data=concrete_data,
                                         neighborhood=neighborhood, casi_min=casi_min, casi_max=casi_max)
        _, shadow_ratio = self.load_shadow_map(neighborhood, data_set_for_shadowing)

        cyclegan_shadow_func = lambda inp: (construct_cyclegan_inference_graph_randomized(inp))
        cyclegan_shadow_op_creater = create_generator_restorer
        cyclegan_shadow_op_initializer = lambda restorer, session: (
            restorer.restore(session, self.get_model_base_dir() + 'shadow_cycle_gan/modelv2/model.ckpt-5668'))

        simple_shadow_func = lambda inp: (construct_simple_shadow_inference_graph(inp, shadow_ratio))
        shadow_dict = {'cycle_gan': ShadowOperationStruct(shadow_op=cyclegan_shadow_func,
                                                          shadow_op_creater=cyclegan_shadow_op_creater,
                                                          shadow_op_initializer=cyclegan_shadow_op_initializer),
                       'simple': ShadowOperationStruct(shadow_op=simple_shadow_func,
                                                       shadow_op_creater=lambda: None,
                                                       shadow_op_initializer=lambda restorer, session: None)}
        return DataSet(shadow_creator_dict=shadow_dict, concrete_data=concrete_data, neighborhood=neighborhood,
                       casi_min=casi_min, casi_max=casi_max)

    def load_shadow_map(self, neighborhood, data_set):
        shadow_map = imread(self.get_model_base_dir() + 'shadow_map.tif')
        shadow_map = numpy.pad(shadow_map, neighborhood, mode='symmetric')
        shadow_ratio = None
        if data_set is not None:
            shadow_ratio = self.calculate_shadow_ratio(data_set.concrete_data[:, :, 0:data_set.concrete_data.shape[2] - 1],
                                                       shadow_map, numpy.logical_not(shadow_map).astype(int))
            shadow_ratio = numpy.append(shadow_ratio, [1]).astype(numpy.float32)
        return shadow_map, shadow_ratio

    def load_samples(self, test_data_ratio):
        train_set = self.read_targets('2013_IEEE_GRSS_DF_Contest_Samples_TR.tif')
        validation_set = self.read_targets('2013_IEEE_GRSS_DF_Contest_Samples_VA.tif')

        # Empty set for 0 ratio for testing
        test_set = numpy.empty([0, train_set.shape[1]])
        if test_data_ratio > 0:
            shuffler = StratifiedShuffleSplit(n_splits=1, test_size=test_data_ratio, random_state=0)
            for train_index, test_index in shuffler.split(train_set[:, 0:1], train_set[:, 2]):
                test_set = train_set[test_index]
                train_set = train_set[train_index]

        return SampleSet(training_targets=train_set, test_targets=test_set,
                         validation_targets=validation_set)

    def read_targets(self, target_image_path):
        targets = imread(self.get_model_base_dir() + target_image_path)
        result = numpy.array([], dtype=int).reshape(0, 3)
        class_range = self.get_class_count()
        for target_index in class_range:
            target_locations = numpy.where(targets == target_index)
            target_locations_as_array = numpy.transpose(
                numpy.vstack((target_locations[1].astype(int), target_locations[0].astype(int))))
            target_index_as_array = numpy.full((len(target_locations_as_array), 1), target_index)
            result = numpy.vstack([result, numpy.hstack((target_locations_as_array, target_index_as_array))])
        return result

    def get_class_count(self):
        return range(0, 15)

    def get_model_base_dir(self):
        return self.base_dir + '/2013_DFTC/'

    def get_data_shape(self, data_set):
        dim = data_set.neighborhood * 2 + 1
        return [dim, dim, data_set.concrete_data.shape[2]]

    def get_scene_shape(self, data_set):
        padding = data_set.neighborhood * 2
        return [data_set.concrete_data.shape[0] - padding, data_set.concrete_data.shape[1] - padding]

    def get_point_value(self, data_set, point):
        return self.__assign_func(data_set.concrete_data, data_set.neighborhood, point)

    @staticmethod
    @jit(nopython=True)
    def __assign_func(concrete_data, neighborhood, point):
        im_pos_x = point[0] + neighborhood  # add as delta due to padding
        im_pos_y = point[1] + neighborhood  # add as delta due to padding
        start_y = im_pos_y - neighborhood
        end_y = start_y + (2 * neighborhood) + 1
        start_x = im_pos_x - neighborhood
        end_x = start_x + (2 * neighborhood) + 1
        value = concrete_data[start_y:end_y:1, start_x:end_x:1, :]
        return value

    def get_target_color_list(self):
        color_list = numpy.zeros([15, 3], numpy.uint8)
        # Grass Healthy
        color_list[0, :] = [0, 180, 0]
        # Grass Stressed
        color_list[1, :] = [0, 124, 0]
        # Grass Synthetic
        color_list[2, :] = [0, 137, 69]
        # Tree
        color_list[3, :] = [0, 69, 0]
        # Soil
        color_list[4, :] = [172, 125, 11]
        # Water
        color_list[5, :] = [0, 190, 194]
        # Residential
        color_list[6, :] = [120, 0, 0]
        # Commercial
        color_list[7, :] = [216, 217, 247]
        # Road
        color_list[8, :] = [121, 121, 121]
        # Highway
        color_list[9, :] = [205, 172, 127]
        # Railway
        color_list[10, :] = [220, 175, 120]
        # Parking lot 1
        color_list[11, :] = [100, 100, 100]
        # Parking lot 2
        color_list[12, :] = [185, 175, 94]
        # Tennis lot
        color_list[13, :] = [0, 237, 0]
        # Running track
        color_list[14, :] = [207, 18, 56]
        return color_list

    @staticmethod
    def calculate_shadow_ratio(casi, shadow_map, shadow_map_inverse):
        shadow_map_inverse_mean = (shadow_map_inverse / numpy.sum(shadow_map_inverse))
        shadow_map_mean = (shadow_map / numpy.sum(shadow_map))

        shadow_map_mean = \
            numpy.repeat(numpy.expand_dims(shadow_map_mean, axis=2), repeats=casi.shape[2], axis=2)
        shadow_map_inverse_mean = \
            numpy.repeat(numpy.expand_dims(shadow_map_inverse_mean, axis=2), repeats=casi.shape[2], axis=2)

        ratio_per_band = numpy.sum(casi * shadow_map_inverse_mean, axis=(0, 1)) / numpy.sum(casi * shadow_map_mean,
                                                                                            axis=(0, 1))
        return ratio_per_band.astype(numpy.float32)

    @staticmethod
    def get_targetbased_shadowed_normal_data(data_set, loader, shadow_map, samples):
        if samples.test_targets is None and samples.training_targets is not None:
            all_targets = samples.training_targets
        elif samples.training_targets is None and samples.test_targets is not None:
            all_targets = samples.test_targets
        else:
            all_targets = numpy.vstack([samples.test_targets, samples.training_targets])

        # First Pass for target size map creation
        class_count = loader.get_class_count().stop
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
        data_shape = loader.get_data_shape(data_set)
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
        return normal_data_as_matrix, shadow_data_as_matrix
