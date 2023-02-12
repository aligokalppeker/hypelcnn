import random
from functools import partial

import numpy
from tifffile import imread

from common.common_nn_ops import INVALID_TARGET_VALUE, shuffle_training_data_using_ratio, \
    shuffle_training_data_using_size, load_shadow_map_common, DataSet, BasicDataSet
from gan.gan_utilities import create_gan_struct, create_simple_shadow_struct
from gan.shadow_data_models import shadowdata_generator_model
from gan.wrappers.cycle_gan_wrapper import CycleGANInferenceWrapper
from loader.DataLoader import SampleSet, LoadingMode
from loader.GULFPORTDataLoader import GULFPORTDataLoader


class MultiDataSet(DataSet):
    _data_sets: tuple[BasicDataSet, ...]
    _primary_data_set: BasicDataSet

    def __init__(self, *data_sets: BasicDataSet) -> None:
        super().__init__()
        self._data_sets = data_sets
        self._primary_data_set = data_sets[0]
        self.lidar = self._primary_data_set.lidar
        self.casi = self._primary_data_set.casi

    def get_data_shape(self):
        return self._primary_data_set.get_data_shape()

    def get_casi_band_count(self):
        return self._primary_data_set.get_casi_band_count()

    def get_scene_shape(self):
        return self._primary_data_set.get_scene_shape()

    def get_unnormalized_casi_dtype(self):
        return self._primary_data_set.get_unnormalized_casi_dtype()

    def get_data_point(self, point_x, point_y):
        selected_idx: int = random.randint(0, len(self._data_sets) - 1)
        return self._data_sets[selected_idx].get_data_point(point_x=point_x, point_y=point_y)


class GULFPORTALTDataLoader(GULFPORTDataLoader):
    _load_mode: LoadingMode

    def __init__(self, base_dir):
        super().__init__(base_dir)
        self._load_mode = LoadingMode.ORIGINAL

    def load_data(self, neighborhood, normalize):
        def _load_original_data():
            return self._load_data_utility(self._hsi_file + self._file_ext,
                                           self._lidar_file + self._file_ext,
                                           neighborhood, normalize)

        def _load_augmented_data(load_mode_val):
            return self._load_data_utility(self._hsi_file + "_" + load_mode_val + self._file_ext,
                                           self._lidar_file + self._file_ext,
                                           neighborhood, normalize)

        if self._load_mode is LoadingMode.ORIGINAL:
            data_set = _load_original_data()
        elif self._load_mode is LoadingMode.SHADOWED or self._load_mode is LoadingMode.DESHADOWED:
            data_set = _load_augmented_data(self._load_mode.value)
        elif self._load_mode is LoadingMode.MIXED:
            original_data_inst = _load_original_data()
            shadowed_data_inst = _load_augmented_data(LoadingMode.SHADOWED.value)
            deshadowed_data_inst = _load_augmented_data(LoadingMode.DESHADOWED.value)
            data_set = MultiDataSet(original_data_inst,
                                    shadowed_data_inst,
                                    deshadowed_data_inst)
        else:
            data_set = _load_original_data()

        _, shadow_ratio = self.load_shadow_map(neighborhood, data_set)
        generator_fn = partial(shadowdata_generator_model, create_only_encoder=False, is_training=False)
        data_set.shadow_creator_dict = {
            "cycle_gan": create_gan_struct(CycleGANInferenceWrapper(generator_fn), self.get_model_base_dir(),
                                           "shadow_gen_model/cycle_gan/model.ckpt-3000"),
            "dcl_gan": create_gan_struct(CycleGANInferenceWrapper(generator_fn), self.get_model_base_dir(),
                                         "shadow_gen_model/dcl_gan/model.ckpt-3000"),
            "dcl_cycle_gan": create_gan_struct(CycleGANInferenceWrapper(generator_fn), self.get_model_base_dir(),
                                               "shadow_gen_model/dcl_cycle_gan/model.ckpt-11652"),
            "simple": create_simple_shadow_struct(shadow_ratio)}
        return data_set

    def load_samples(self, train_data_ratio, test_data_ratio):
        shadow_map, _ = self.load_shadow_map(0, None)

        targets = imread(self.get_model_base_dir() + 'muulf_gt_shadow_corrected.tif')

        targets_with_shadow = numpy.copy(targets)
        targets_with_shadow[numpy.logical_not(shadow_map)] = INVALID_TARGET_VALUE
        result_with_shadow = self._convert_targets_aux(targets_with_shadow)

        targets_in_clear_area = numpy.copy(targets)
        targets_in_clear_area[shadow_map.astype(bool)] = INVALID_TARGET_VALUE
        result_in_clear_area = self._convert_targets_aux(targets_in_clear_area)

        if train_data_ratio < 1.0:
            train_set, validation_set = shuffle_training_data_using_ratio(result_in_clear_area, train_data_ratio)
        else:
            train_set, validation_set = shuffle_training_data_using_size(self.get_class_count(),
                                                                         result_in_clear_area,
                                                                         int(train_data_ratio),
                                                                         None)
        # Empty set for 0 ratio for testing
        test_set = numpy.empty([0, train_set.shape[1]])

        # Add shadow targets to validation sets
        validation_set = numpy.vstack([validation_set, result_with_shadow])

        return SampleSet(training_targets=train_set, test_targets=test_set, validation_targets=validation_set)

    def load_shadow_map(self, neighborhood, data_set):
        return load_shadow_map_common(data_set, neighborhood, self.get_model_base_dir() + "muulf_shadow_map.tif")
