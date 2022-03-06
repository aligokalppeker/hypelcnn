from collections import namedtuple

import numpy
from numba import jit
from tifffile import imread

from DataLoader import DataLoader, SampleSet
from common_nn_operations import calculate_shadow_ratio, read_targets_from_image, shuffle_test_data_using_ratio
from utilities.gan_utilities import create_simple_shadow_struct, create_gan_struct
from utilities.cycle_gan_wrapper import CycleGANInferenceWrapper

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
            casi_max = numpy.max(casi, axis=(0, 1)).astype(numpy.float32)
            casi = casi / casi_max

        concrete_data = numpy.zeros([casi.shape[0], casi.shape[1], casi.shape[2] + 1], dtype=numpy.float32)
        concrete_data[:, :, 0:concrete_data.shape[2] - 1] = casi
        concrete_data[:, :, concrete_data.shape[2] - 1] = lidar[:, :, 0]

        data_set_for_shadowing = DataSet(shadow_creator_dict=None, concrete_data=concrete_data,
                                         neighborhood=neighborhood, casi_min=casi_min, casi_max=casi_max)
        _, shadow_ratio = self.load_shadow_map(neighborhood, data_set_for_shadowing)

        shadow_dict = {'cycle_gan': create_gan_struct(CycleGANInferenceWrapper(),
                                                      self.get_model_base_dir(),
                                                      "shadow_cycle_gan/modelv2/model.ckpt-5668"),
                       'simple': create_simple_shadow_struct(shadow_ratio)}
        return DataSet(shadow_creator_dict=shadow_dict, concrete_data=concrete_data, neighborhood=neighborhood,
                       casi_min=casi_min, casi_max=casi_max)

    def get_hsi_lidar_data(self, data_set):
        last_index = data_set.concrete_data.shape[2] - 1
        return data_set.concrete_data[:, :, 0:last_index], data_set.concrete_data[:, :, last_index]

    def load_shadow_map(self, neighborhood, data_set):
        shadow_map = imread(self.get_model_base_dir() + 'shadow_map.tif')
        shadow_map = numpy.pad(shadow_map, neighborhood, mode='symmetric')
        shadow_ratio = None
        if data_set is not None:
            shadow_ratio = calculate_shadow_ratio(
                data_set.concrete_data[:, :, 0:data_set.concrete_data.shape[2] - 1],
                shadow_map, numpy.logical_not(shadow_map).astype(int))
            shadow_ratio = numpy.append(shadow_ratio, [1]).astype(numpy.float32)
        return shadow_map, shadow_ratio

    def load_samples(self, train_data_ratio, test_data_ratio):
        train_set = self.read_targets('2013_IEEE_GRSS_DF_Contest_Samples_TR.tif')
        validation_set = self.read_targets('2013_IEEE_GRSS_DF_Contest_Samples_VA.tif')

        test_set, train_set = shuffle_test_data_using_ratio(train_set, test_data_ratio)

        return SampleSet(training_targets=train_set, test_targets=test_set,
                         validation_targets=validation_set)

    def read_targets(self, target_image_path):
        targets = imread(self.get_model_base_dir() + target_image_path)
        return read_targets_from_image(targets, self.get_class_count())

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

    def get_band_measurements(self):
        return numpy.linspace(380, 1050, num=144)
