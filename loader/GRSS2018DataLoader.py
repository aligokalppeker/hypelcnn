import numpy
from numba import jit
from tifffile import imread

from loader.DataLoader import DataLoader, SampleSet
from common_nn_operations import shuffle_test_data_using_ratio, shuffle_training_data_using_ratio, \
    shuffle_training_data_using_size, DataSet


class GRSS2018DataLoader(DataLoader):

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, neighborhood, normalize):
        casi = imread(self.get_model_base_dir() + "20170218_UH_CASI_S4_NAD83.tiff")[:, :, 0:-2]
        lidar = imread(self.get_model_base_dir() + "UH17c_GEF051.tif")[:, :, numpy.newaxis]
        lidar[numpy.where(lidar > 300)] = 0  # Eliminate unacceptable values
        return DataSet(shadow_creator_dict=None, casi=casi, lidar=lidar, neighborhood=neighborhood, normalize=normalize)

    @staticmethod
    def print_stats(data):
        for band_index in range(1, data.shape[2]):
            band_data = data[:, :, band_index]
            print('Band mean:%.5f, band std:%.5f, min:%.5f, max:%.5f' % (
                numpy.mean(band_data), numpy.std(band_data), numpy.min(band_data), numpy.max(band_data)))

    def load_samples(self, train_data_ratio, test_data_ratio):
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

        if train_data_ratio < 1.0:
            train_set, validation_set = shuffle_training_data_using_ratio(result, train_data_ratio)
        else:
            train_data_ratio = int(train_data_ratio)
            train_set, validation_set = shuffle_training_data_using_size(self.get_class_count(),
                                                                         result,
                                                                         train_data_ratio,
                                                                         None)

        test_set, train_set = shuffle_test_data_using_ratio(train_set, test_data_ratio)

        return SampleSet(training_targets=train_set, test_targets=test_set, validation_targets=validation_set)

    def load_shadow_map(self, neighborhood, data_set):
        pass

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

    def get_point_value(self, data_set, point):
        neighborhood = data_set.neighborhood
        size = neighborhood * 2 + 1
        total_band_count = data_set.casi.shape[2] + 1
        result = numpy.empty([size, size, total_band_count], dtype=data_set.casi.dtype)

        start_x, start_y = self.__calculate_position(neighborhood, point, 0.5)
        lidar_start_x, lidar_start_y = self.__calculate_position(neighborhood, point, 1)

        self.__assign_loop(data_set.casi, data_set.lidar,
                           lidar_start_x, lidar_start_y, start_x, start_y,
                           size, total_band_count, result)
        return result

    @staticmethod
    @jit(nopython=True)
    def __assign_loop(casi, lidar, lidar_start_x, lidar_start_y, start_x, start_y, size, total_band_count, result):
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

    def get_band_measurements(self):
        return numpy.linspace(380, 1050, num=48)
