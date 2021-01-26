from collections import namedtuple

import numpy
from sklearn.model_selection import StratifiedShuffleSplit
from tifffile import imread

from DataLoader import DataLoader, SampleSet

DataSet = namedtuple('DataSet', ['shadow_creator_dict', 'casi', 'lidar', 'neighborhood', 'casi_min', 'casi_max'])


def __assign_func(data_array, neighborhood, point):
    neighborhood_delta = (neighborhood + neighborhood) + 1
    start_y = point[1]  # + neighborhood - neighborhood; add as delta due to padding and windowed back
    end_y = start_y + neighborhood_delta
    start_x = point[0]  # + neighborhood - neighborhood; add as delta due to padding and windowed back
    end_x = start_x + neighborhood_delta
    return data_array[start_y:end_y:1, start_x:end_x:1, :]


def get_point_value_impl(data_set, point):
    casi_patch = __assign_func(data_set.casi, data_set.neighborhood, point)
    lidar_patch = __assign_func(data_set.lidar, data_set.neighborhood, point)
    return numpy.concatenate((casi_patch, lidar_patch), axis=2)


class GULFPORTDataLoader(DataLoader):

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def get_original_data_type(self):
        return numpy.float32

    def load_data(self, neighborhood, normalize):
        casi = imread(self.get_model_base_dir() + 'muulf_hsi.tif')
        lidar = numpy.expand_dims(imread(self.get_model_base_dir() + 'muulf_lidar.tif'), axis=2)

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

        return DataSet(shadow_creator_dict=None, casi=casi, lidar=lidar, neighborhood=neighborhood,
                       casi_min=casi_min, casi_max=casi_max)

    def load_samples(self, test_data_ratio):
        result = self._convert_samplemap_to_array(imread(self.get_model_base_dir() + 'muulf_gt.tif').astype(numpy.int8))

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

    def _convert_samplemap_to_array(self, targets):
        result = numpy.array([], dtype=int).reshape(0, 3)
        for target_index in range(1, 12):
            target_locations = numpy.where(targets == target_index)
            target_locations_as_array = numpy.transpose(
                numpy.vstack((target_locations[1].astype(int), target_locations[0].astype(int))))
            target_index_as_array = numpy.full((len(target_locations_as_array), 1), target_index - 1)  # TargetIdx 0..11
            result = numpy.vstack([result, numpy.hstack((target_locations_as_array, target_index_as_array))])
        return result

    def load_shadow_map(self, neighborhood, data_set):
        pass

    def get_class_count(self):
        return range(0, 11)

    def get_target_color_list(self):
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
        return self.base_dir + '/GULFPORT/'

    def get_data_shape(self, data_set):
        dim = data_set.neighborhood * 2 + 1
        return [dim, dim, (data_set.casi.shape[2] + 1)]

    def get_scene_shape(self, data_set):
        padding = data_set.neighborhood * 2
        return [data_set.lidar.shape[0] - padding, data_set.lidar.shape[1] - padding]

    def get_point_value(self, data_set, point):
        return get_point_value_impl(data_set, point)
