import numpy
from sklearn.model_selection import StratifiedShuffleSplit
from tifffile import imread

from DataLoader import SampleSet
from GULFPORTDataLoader import GULFPORTDataLoader
from common_nn_operations import INVALID_TARGET_VALUE


class GULFPORTALTDataLoader(GULFPORTDataLoader):

    def __init__(self, base_dir):
        super().__init__(base_dir)

    def load_samples(self, test_data_ratio):
        shadow_map = imread(self.get_model_base_dir() + 'muulf_shadow_map.tif')

        targets = imread(self.get_model_base_dir() + 'muulf_gt_shadow_corrected.tif')
        targets_with_shadow = numpy.copy(targets)
        targets_with_shadow[numpy.logical_not(shadow_map)] = INVALID_TARGET_VALUE
        result_with_shadow = self._convert_samplemap_to_array(targets_with_shadow)

        targets_in_clear_area = numpy.copy(targets)
        targets_in_clear_area[shadow_map.astype(bool)] = INVALID_TARGET_VALUE
        result_in_clear_area = self._convert_samplemap_to_array(targets_in_clear_area)

        validation_set = None
        train_set = None
        validation_data_ratio = 0.90
        shuffler = StratifiedShuffleSplit(n_splits=1, test_size=validation_data_ratio)
        for train_index, test_index in shuffler.split(result_in_clear_area[:, 0:1], result_in_clear_area[:, 2]):
            validation_set = result_in_clear_area[test_index]
            train_set = result_in_clear_area[train_index]

        # Empty set for 0 ratio for testing
        test_set = numpy.empty([0, train_set.shape[1]])
        if test_data_ratio > 0:
            train_shuffler = StratifiedShuffleSplit(n_splits=1, test_size=test_data_ratio, random_state=0)
            for train_index, test_index in train_shuffler.split(train_set[:, 0:1], train_set[:, 2]):
                test_set = train_set[test_index]
                train_set = train_set[train_index]

        # Add shadow targets to validation sets
        validation_set = numpy.vstack([validation_set, result_with_shadow])

        return SampleSet(training_targets=train_set, test_targets=test_set, validation_targets=validation_set)
