import numpy
from numba import jit
from tifffile import imread

from loader.DataLoader import DataLoader
from common_nn_operations import DataSet, load_shadow_map_common

BLANK_OFFSET = 55


class AVONDataLoader(DataLoader):

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, neighborhood, normalize):
        casi = imread(self.get_model_base_dir() + "0920-1857.georef_cropped.tif")[:, :, BLANK_OFFSET:-BLANK_OFFSET]
        casi = casi.astype(numpy.uint16)
        casi = numpy.swapaxes(casi, axis1=0, axis2=2)
        outlier_in_upper_bound = numpy.percentile(casi, 95, axis=[0, 1]).astype(casi.dtype)
        numpy.clip(casi, None, outlier_in_upper_bound, out=casi)
        return DataSet(shadow_creator_dict=None, casi=casi, lidar=None, neighborhood=neighborhood, normalize=normalize,
                       casi_min=0)

    def load_shadow_map(self, neighborhood, data_set):
        shadow_map, shadow_ratio = load_shadow_map_common(data_set, neighborhood,
                                                          self.get_model_base_dir() + "0920-1857.georef_cropped_shadow.tif")
        # shadow_map = ndimage.binary_dilation(shadow_map, iterations=1).astype(shadow_map.dtype)
        return shadow_map, shadow_ratio

    def load_samples(self, train_data_ratio, test_data_ratio):
        raise NotImplementedError

    def get_class_count(self):
        raise NotImplementedError

    def get_target_color_list(self):
        raise NotImplementedError

    def get_model_base_dir(self):
        return self.base_dir + '/AVON/'

    def get_point_value(self, data_set, point):
        return self.__assign_func(data_set.casi, data_set.neighborhood, point)

    @staticmethod
    @jit(nopython=True)
    def __assign_func(casi, neighborhood, point):
        start_x = point[0]
        start_y = point[1]
        end_x = start_x + (2 * neighborhood) + 1
        end_y = start_y + (2 * neighborhood) + 1
        return casi[start_y:end_y:1, start_x:end_x:1, :]

    def get_band_measurements(self):
        return numpy.linspace(400, 2500, num=360)
