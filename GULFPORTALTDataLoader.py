import numpy
from tifffile import imread

from DataLoader import SampleSet
from GULFPORTDataLoader import GULFPORTDataLoader, DataSet
from common_nn_operations import INVALID_TARGET_VALUE, calculate_shadow_ratio, shuffle_training_data_using_ratio, \
    shuffle_training_data_using_size
from shadow_data_generator import create_gan_struct, create_simple_shadow_struct
from utilities.cycle_gan_wrapper import CycleGANInferenceWrapper


class GULFPORTALTDataLoader(GULFPORTDataLoader):

    def __init__(self, base_dir):
        super().__init__(base_dir)

    def load_data(self, neighborhood, normalize):
        data_set = super().load_data(neighborhood, normalize)
        _, shadow_ratio = self.load_shadow_map(neighborhood, data_set)
        shadow_dict = {'cycle_gan': create_gan_struct(CycleGANInferenceWrapper(),
                                                      self.get_model_base_dir(),
                                                      "shadow_cycle_gan/dualgan/model.ckpt-49453"),
                       'simple': create_simple_shadow_struct(shadow_ratio)}

        return DataSet(shadow_creator_dict=shadow_dict, casi=data_set.casi, lidar=data_set.lidar,
                       neighborhood=data_set.neighborhood,
                       casi_min=data_set.casi_min, casi_max=data_set.casi_max)

    def load_samples(self, train_data_ratio, test_data_ratio):
        shadow_map, _ = self.load_shadow_map(0, None)

        targets = imread(self.get_model_base_dir() + 'muulf_gt_shadow_corrected.tif')

        targets_with_shadow = numpy.copy(targets)
        targets_with_shadow[numpy.logical_not(shadow_map)] = INVALID_TARGET_VALUE
        result_with_shadow = self._convert_targets_aux(targets_with_shadow)

        targets_in_clear_area = numpy.copy(targets)
        targets_in_clear_area[shadow_map.astype(bool)] = INVALID_TARGET_VALUE
        result_in_clear_area = self._convert_targets_aux(targets_in_clear_area)

        # train_data_ratio = 100
        if train_data_ratio < 1.0:
            train_set, validation_set = shuffle_training_data_using_ratio(result_in_clear_area, train_data_ratio)
        else:
            train_data_ratio = int(train_data_ratio)
            train_set, validation_set = shuffle_training_data_using_size(self.get_class_count(),
                                                                         result_in_clear_area,
                                                                         train_data_ratio,
                                                                         None)
        # Empty set for 0 ratio for testing
        test_set = numpy.empty([0, train_set.shape[1]])

        # Add shadow targets to validation sets
        validation_set = numpy.vstack([validation_set, result_with_shadow])

        return SampleSet(training_targets=train_set, test_targets=test_set, validation_targets=validation_set)

    def load_shadow_map(self, neighborhood, data_set):
        shadow_map = imread(self.get_model_base_dir() + 'muulf_shadow_map.tif')
        shadow_map = numpy.pad(shadow_map, neighborhood, mode='symmetric')
        shadow_ratio = None
        if data_set is not None:
            shadow_ratio = calculate_shadow_ratio(data_set.casi,
                                                  shadow_map,
                                                  numpy.logical_not(shadow_map).astype(int))
            shadow_ratio = numpy.append(shadow_ratio, [1]).astype(numpy.float32)
        return shadow_map, shadow_ratio
