from collections import namedtuple

import numpy
from numba import jit
from sklearn.model_selection import StratifiedShuffleSplit
from tifffile import imread

from DataLoader import DataLoader, SampleSet

DataSet = namedtuple('DataSet', ['shadow_creator_dict', 'casi', 'lidar', 'neighborhood', 'casi_min', 'casi_max'])


class GRSS2018DataLoader(DataLoader):

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, neighborhood, normalize):
        casi = imread(self.get_model_base_dir() + '20170218_UH_CASI_S4_NAD83.tiff')[:, :, 0:-2]
        lidar = imread(self.get_model_base_dir() + 'UH17c_GEF051.tif')[:, :, numpy.newaxis]
        lidar[numpy.where(lidar > 300)] = 0  # Eliminate unacceptable values
        # lidar = downscale_local_mean(lidar, (2, 2, 1))

        # Padding part
        pad_size = ((neighborhood, neighborhood), (neighborhood, neighborhood), (0, 0))
        lidar = numpy.pad(lidar, pad_size, mode='symmetric')
        casi = numpy.pad(casi, pad_size, mode='symmetric')  # Half the pad ???

        casi_min = None
        casi_max = None
        if normalize:
            # Normalization
            lidar -= numpy.min(lidar)
            lidar = lidar / numpy.max(lidar)
            casi_min = numpy.min(casi, axis=(0, 1))
            casi -= casi_min
            casi_max = numpy.max(casi, axis=(0, 1))
            casi = casi / casi_max.astype(numpy.float32)

        data_set = DataSet(shadow_creator_dict=None, casi=casi, lidar=lidar, neighborhood=neighborhood,
                           casi_min=casi_min, casi_max=casi_max)

        return data_set

    @staticmethod
    def print_stats(data):
        for band_index in range(1, data.shape[2]):
            band_data = data[:, :, band_index]
            print('Band mean:%.5f, band std:%.5f, min:%.5f, max:%.5f' % (
                numpy.mean(band_data), numpy.std(band_data), numpy.min(band_data), numpy.max(band_data)))

    def load_samples(self, test_data_ratio):
        targets = imread(self.get_model_base_dir() + '2018_IEEE_GRSS_DFC_GT_TR.tif')
        result = numpy.array([], dtype=int).reshape(0, 3)
        y_delta = 1202
        x_delta = 1194
        for target_index in range(1, 21):
            target_locations = numpy.where(targets == target_index)
            target_locations_as_array = numpy.transpose(
                numpy.vstack((target_locations[1].astype(int) + x_delta, target_locations[0].astype(int) + y_delta)))
            target_index_as_array = numpy.full((len(target_locations_as_array), 1), target_index - 1)  # TargetIdx 0..20
            result = numpy.vstack([result, numpy.hstack((target_locations_as_array, target_index_as_array))])

        validation_data_ratio = 0.90
        shuffler = StratifiedShuffleSplit(n_splits=1, test_size=validation_data_ratio)
        for train_index, test_index in shuffler.split(result[:, 0:1], result[:, 2]):
            validation_set = result[test_index]
            train_set = result[train_index]

        # Empty set for 0 ratio for testing
        test_set = numpy.empty([0, train_set.shape[1]])
        if test_data_ratio > 0:
            train_shuffler = StratifiedShuffleSplit(n_splits=1, test_size=test_data_ratio, random_state=0)
            for train_index, test_index in train_shuffler.split(train_set[:, 0:1], train_set[:, 2]):
                test_set = train_set[test_index]
                train_set = train_set[train_index]

        return SampleSet(training_targets=train_set, test_targets=test_set, validation_targets=validation_set)

    def get_class_count(self):
        return range(0, 20)

    def get_target_color_list(self):
        color_list = numpy.zeros([21, 3], numpy.uint8)
        # No target
        # color_list[0, :] = [0, 0, 0]
        # Grass Healthy
        color_list[0, :] = [0, 180, 0]
        # Grass Stressed
        color_list[1, :] = [0, 124, 0]
        # Grass Synthetic, Artificial Turf
        color_list[2, :] = [0, 137, 69]
        # Evergreen Tree
        color_list[3, :] = [0, 69, 0]
        # Decidious Tree, self assigned
        color_list[4, :] = [255, 0, 0]
        # Soil, Bare Earth
        color_list[5, :] = [172, 125, 11]
        # Water
        color_list[6, :] = [0, 190, 194]
        # Residential buildings
        color_list[7, :] = [120, 0, 0]
        # Commercial, Non residential buildings
        color_list[8, :] = [216, 217, 247]
        # Road
        color_list[9, :] = [121, 121, 121]
        # Sidewalks, self assigned
        color_list[10, :] = [255, 255, 0]
        # Crosswalks, self assigned
        color_list[11, :] = [0, 155, 50]
        # Major thoroughfares, self assigned
        color_list[12, :] = [0, 55, 55]
        # Highway
        color_list[13, :] = [205, 172, 127]
        # Railway
        color_list[14, :] = [220, 175, 120]
        # Parking lot 1, paved parking lots
        color_list[15, :] = [100, 100, 100]
        # Parking lot 2, unpaved parking lots
        color_list[16, :] = [185, 175, 94]
        # Cars
        color_list[17, :] = [0, 237, 0]
        # Trains
        color_list[18, :] = [207, 18, 56]
        # Stadium Seats, self assigned
        color_list[19, :] = [0, 0, 255]
        return color_list

    def get_model_base_dir(self):
        return self.base_dir + '/2018_DFTC/'

    def get_data_shape(self, data_set):
        dim = data_set.neighborhood * 2 + 1
        return [dim, dim, (data_set.casi.shape[2] + 1)]

    def get_scene_shape(self, data_set):
        padding = data_set.neighborhood * 2
        return [data_set.lidar.shape[0] - padding, data_set.lidar.shape[1] - padding]

    def get_point_value(self, data_set, point):
        neighborhood = data_set.neighborhood
        size = neighborhood * 2 + 1
        total_band_count = data_set.casi.shape[2] + 1
        result = numpy.empty([size, size, total_band_count], dtype=data_set.casi.dtype)

        start_x, start_y = self.__calculate_position(neighborhood, point, 0.5)
        lidar_start_x, lidar_start_y = self.__calculate_position(neighborhood, point, 1)

        self.__assign_loop(data_set, lidar_start_x, lidar_start_y, start_x, start_y,
                           size, total_band_count, result)
        return result

    @staticmethod
    @jit(nopython=True)
    def __assign_loop(data_set, lidar_start_x, lidar_start_y, start_x, start_y, size, total_band_count, result):
        casi = data_set.casi
        lidar = data_set.lidar
        last_element_index = total_band_count - 1
        for x_index in range(0, size):
            for y_index in range(0, size):
                result[y_index, x_index, 0:last_element_index] = \
                    casi[start_y + int(y_index * 0.5), start_x + int(x_index * 0.5), :]
                result[y_index, x_index, last_element_index] = \
                    lidar[lidar_start_y + y_index, lidar_start_x + x_index, 0]

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __calculate_position(neighborhood, point, scale):
        actual_padding = int(neighborhood * scale)
        start_y = int(point[1] * scale) + neighborhood - actual_padding  # delta padding - actual padding
        start_x = int(point[0] * scale) + neighborhood - actual_padding  # delta padding - actual padding
        return start_x, start_y
