import numpy
from tifffile import imread

from common.common_nn_ops import read_targets_from_image, shuffle_training_data_using_ratio, \
    shuffle_training_data_using_size, shuffle_test_data_using_ratio, BasicDataSet
from loader.DataLoader import DataLoader, SampleSet


class GULFPORTDataLoader(DataLoader):

    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._hsi_file = "muulf_hsi"
        self._lidar_file = "muulf_lidar"
        self._file_ext = ".tif"

    def load_data(self, neighborhood, normalize):
        return self._load_data_utility(self._hsi_file + self._file_ext, self._lidar_file + self._file_ext, neighborhood,
                                       normalize)

    def _load_data_utility(self, hsi_file, lidar_file, neighborhood, normalize, casi_min=None, casi_max=None):
        casi = imread(self.get_model_base_dir() + hsi_file)
        lidar = numpy.expand_dims(imread(self.get_model_base_dir() + lidar_file), axis=2)
        return BasicDataSet(shadow_creator_dict=None, casi=casi, lidar=lidar, neighborhood=neighborhood,
                            normalize=normalize, casi_min=casi_min, casi_max=casi_max)

    def load_samples(self, train_data_ratio, test_data_ratio):
        result = self.read_targets("muulf_gt.tif")

        if train_data_ratio < 1.0:
            train_set, validation_set = shuffle_training_data_using_ratio(result, train_data_ratio)
        else:
            train_set, validation_set = shuffle_training_data_using_size(self.get_class_count(),
                                                                         result,
                                                                         int(train_data_ratio),
                                                                         None)

        test_set, train_set = shuffle_test_data_using_ratio(train_set, test_data_ratio)

        return SampleSet(training_targets=train_set, test_targets=test_set, validation_targets=validation_set)

    def read_targets(self, target_image_path):
        targets = imread(self.get_model_base_dir() + target_image_path)
        return self._convert_targets_aux(targets)

    @staticmethod
    def _convert_targets_aux(targets):
        return read_targets_from_image(targets, range(1, 12)) - [0, 0, 1]

    def load_shadow_map(self, neighborhood, data_set):
        pass

    def get_class_count(self):
        return range(0, 11)

    def get_samples_color_list(self):
        color_list = numpy.zeros([11, 3], numpy.uint8)
        # No target
        # color_list[0, :] = [0, 0, 0]
        # 'trees'
        color_list[0, :] = [0, 128, 0]
        # 'grass_pure'
        color_list[1, :] = [25, 255, 25]
        # 'grass_groundsurface'
        color_list[2, :] = [0, 255, 255]
        # 'dirt_and_sand'
        color_list[3, :] = [255, 204, 0]
        # 'road_materials'
        color_list[4, :] = [255, 20, 67]
        # 'water'
        color_list[5, :] = [0, 0, 204]
        # 'shadow_building'
        color_list[6, :] = [102, 0, 204]
        # 'buildings'
        color_list[7, :] = [255, 132, 156]
        # 'sidewalk'
        color_list[8, :] = [204, 102, 0]
        # 'yellowcurb'
        color_list[9, :] = [255, 255, 207]
        # 'cloth_panels'
        color_list[10, :] = [208, 45, 115]
        return color_list

    def get_model_base_dir(self):
        return self._base_dir + '/GULFPORT/'

    def get_band_measurements(self):
        return numpy.linspace(405, 1005, 64)
