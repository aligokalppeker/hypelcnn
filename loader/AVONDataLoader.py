from functools import partial

import numpy
from numba import jit

from gan.gan_utilities import create_gan_struct, create_simple_shadow_struct
from gan.shadow_data_models import shadowdata_generator_model
from gan.wrappers.cycle_gan_wrapper import CycleGANInferenceWrapper
from loader.DataLoader import DataLoader, SampleSet
from common.common_nn_ops import DataSet, load_shadow_map_common, shuffle_test_data_using_ratio, \
    read_targets_from_image, shuffle_training_data_using_size

BLANK_OFFSET = 55


class AVONDataLoader(DataLoader):

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, neighborhood, normalize):
        from tifffile import imread
        casi = imread(self.get_model_base_dir() + "0920-1857.georef_cropped.tif")[:, :, BLANK_OFFSET:-BLANK_OFFSET]
        casi = casi.astype(numpy.uint16)
        casi = numpy.swapaxes(casi, axis1=0, axis2=2)
        outlier_in_upper_bound = numpy.percentile(casi, 95, axis=[0, 1]).astype(casi.dtype)
        numpy.clip(casi, None, outlier_in_upper_bound, out=casi)
        data_set = DataSet(shadow_creator_dict=None, casi=casi, lidar=None, neighborhood=neighborhood,
                           normalize=normalize,
                           casi_min=0)

        _, shadow_ratio = self.load_shadow_map(neighborhood, data_set)
        generator_fn = partial(shadowdata_generator_model, create_only_encoder=False, is_training=False)
        data_set.shadow_creator_dict = {
            "cycle_gan": create_gan_struct(CycleGANInferenceWrapper(generator_fn), self.get_model_base_dir(),
                                           "shadow_gen_model/cycle_gan/model.ckpt-7000"),
            "dcl_gan": create_gan_struct(CycleGANInferenceWrapper(generator_fn), self.get_model_base_dir(),
                                         "shadow_gen_model/dcl_gan/model.ckpt-6000"),
            "dcl_cycle_gan": create_gan_struct(CycleGANInferenceWrapper(generator_fn), self.get_model_base_dir(),
                                               "shadow_gen_model/dcl_cycle_gan/model.ckpt-3000"),
            "simple": create_simple_shadow_struct(shadow_ratio)}

        return data_set

    def load_shadow_map(self, neighborhood, data_set):
        shadow_map, shadow_ratio = load_shadow_map_common(data_set, neighborhood,
                                                          self.get_model_base_dir() + "0920-1857.georef_cropped_shadow.tif")
        # shadow_map = ndimage.binary_dilation(shadow_map, iterations=1).astype(shadow_map.dtype)
        return shadow_map, shadow_ratio

    def load_samples(self, train_data_ratio, test_data_ratio):
        non_shadow_set_t1 = self.read_targets("0920-1857.georef_cropped_rgb_with_targets_1_nsh.bmp", target_no=1)
        shadow_set_shadow_t1 = self.read_targets("0920-1857.georef_cropped_rgb_with_targets_1_sh.bmp", target_no=1)
        non_shadow_set_t2 = self.read_targets("0920-1857.georef_cropped_rgb_with_targets_2_nsh.bmp", target_no=2)
        shadow_set_shadow_t2 = self.read_targets("0920-1857.georef_cropped_rgb_with_targets_2_sh.bmp", target_no=2)

        if train_data_ratio < 1.0:
            train_set_t1, validation_set_t1 = shuffle_test_data_using_ratio(non_shadow_set_t1, train_data_ratio)
            train_set_t2, validation_set_t2 = shuffle_test_data_using_ratio(non_shadow_set_t2, train_data_ratio)
        else:
            train_set_t1, validation_set_t1 = shuffle_training_data_using_size(self.get_class_count(),
                                                                               non_shadow_set_t1,
                                                                               int(train_data_ratio),
                                                                               None)
            train_set_t2, validation_set_t2 = shuffle_training_data_using_size(self.get_class_count(),
                                                                               non_shadow_set_t2,
                                                                               int(train_data_ratio),
                                                                               None)
        train_set = numpy.vstack([train_set_t1, train_set_t2])
        validation_set = numpy.vstack(
            [shadow_set_shadow_t1, shadow_set_shadow_t2, validation_set_t1, validation_set_t2])

        test_set, train_set = shuffle_test_data_using_ratio(train_set, test_data_ratio)

        return SampleSet(training_targets=train_set, test_targets=test_set,
                         validation_targets=validation_set)

    def read_targets(self, target_image_path, target_no):
        from imageio.v2 import imread
        image = imread(self.get_model_base_dir() + target_image_path)[BLANK_OFFSET:-BLANK_OFFSET, :]
        targets = ((image / 255).astype(int) * target_no) - 1
        return read_targets_from_image(targets, self.get_class_count())

    def get_class_count(self):
        return range(0, 2)

    def get_samples_color_list(self):
        color_list = numpy.zeros([1, 3], numpy.uint8)
        # Class 1
        color_list[0, :] = [0, 0, 255]
        # Class 2
        color_list[0, :] = [255, 0, 0]

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
