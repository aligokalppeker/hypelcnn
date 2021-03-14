import numpy
from sklearn.model_selection import StratifiedShuffleSplit
from tifffile import imread

from DataLoader import SampleSet, ShadowOperationStruct
from GULFPORTDataLoader import GULFPORTDataLoader, DataSet
from common_nn_operations import INVALID_TARGET_VALUE, calculate_shadow_ratio
from shadow_data_generator import construct_simple_shadow_inference_graph, create_generator_restorer, \
    construct_cyclegan_inference_graph_randomized


class GULFPORTALTDataLoader(GULFPORTDataLoader):

    def __init__(self, base_dir):
        super().__init__(base_dir)

    def load_data(self, neighborhood, normalize):
        data_set = super().load_data(neighborhood, normalize)
        shadow_map, shadow_ratio = self.load_shadow_map(neighborhood, data_set)

        # Add extra lidar ratio as 1
        shadow_ratio = numpy.append(shadow_ratio, [1]).astype(numpy.float32)

        cyclegan_shadow_func = lambda inp: (construct_cyclegan_inference_graph_randomized(inp))
        cyclegan_shadow_op_creater = create_generator_restorer
        cyclegan_shadow_op_initializer = lambda restorer, session: (
            restorer.restore(session, self.get_model_base_dir() + 'shadow_cycle_gan/v1/model.ckpt-22519'))

        simple_shadow_func = lambda inp: (construct_simple_shadow_inference_graph(inp, shadow_ratio))
        shadow_dict = {'cycle_gan': ShadowOperationStruct(shadow_op=cyclegan_shadow_func,
                                                          shadow_op_creater=cyclegan_shadow_op_creater,
                                                          shadow_op_initializer=cyclegan_shadow_op_initializer),
                       'simple': ShadowOperationStruct(shadow_op=simple_shadow_func,
                                                       shadow_op_creater=lambda: None,
                                                       shadow_op_initializer=lambda restorer, session: None)}

        return DataSet(shadow_creator_dict=shadow_dict, casi=data_set.casi, lidar=data_set.lidar,
                       neighborhood=data_set.neighborhood,
                       casi_min=data_set.casi_min, casi_max=data_set.casi_max)

    def load_samples(self, test_data_ratio):
        shadow_map, _ = self.load_shadow_map(0, None)

        targets = imread(self.get_model_base_dir() + 'muulf_gt_shadow_corrected.tif')

        targets_with_shadow = numpy.copy(targets)
        targets_with_shadow[numpy.logical_not(shadow_map)] = INVALID_TARGET_VALUE
        result_with_shadow = self._convert_targets_aux(targets_with_shadow)

        targets_in_clear_area = numpy.copy(targets)
        targets_in_clear_area[shadow_map.astype(bool)] = INVALID_TARGET_VALUE
        result_in_clear_area = self._convert_targets_aux(targets_in_clear_area)

        train_data_ratio = 100
        validation_data_ratio = None

        if isinstance(train_data_ratio, float):
            train_set, validation_set = self.shuffle_training_data_using_ratio(result_in_clear_area, train_data_ratio)
        else:
            train_set, validation_set = self.shuffle_training_data_using_size(result_in_clear_area, train_data_ratio,
                                                                              validation_data_ratio)
        # Empty set for 0 ratio for testing
        test_set = numpy.empty([0, train_set.shape[1]])

        # Add shadow targets to validation sets
        validation_set = numpy.vstack([validation_set, result_with_shadow])

        return SampleSet(training_targets=train_set, test_targets=test_set, validation_targets=validation_set)

    @staticmethod
    def shuffle_training_data_using_ratio(result_in_clear_area, train_data_ratio):
        validation_set = None
        train_set = None
        shuffler = StratifiedShuffleSplit(n_splits=1, train_size=train_data_ratio)
        for train_index, test_index in shuffler.split(result_in_clear_area[:, 0:1], result_in_clear_area[:, 2]):
            validation_set = result_in_clear_area[test_index]
            train_set = result_in_clear_area[train_index]
        return train_set, validation_set

    def load_shadow_map(self, neighborhood, data_set):
        shadow_map = imread(self.get_model_base_dir() + 'muulf_shadow_map.tif')
        shadow_map = numpy.pad(shadow_map, neighborhood, mode='symmetric')
        shadow_ratio = None
        if data_set is not None:
            shadow_ratio = calculate_shadow_ratio(data_set.casi,
                                                  shadow_map,
                                                  numpy.logical_not(shadow_map).astype(int))
        return shadow_map, shadow_ratio

    def shuffle_training_data_using_size(self, result_in_clear_area, train_data_size, validation_size):
        sample_id_list = result_in_clear_area[:, 2]
        train_set = numpy.empty([0, result_in_clear_area.shape[1]], dtype=numpy.int)
        validation_set = numpy.empty([0, result_in_clear_area.shape[1]], dtype=numpy.int)
        for sample_class in self.get_class_count():
            id_for_class = numpy.where(sample_id_list == sample_class)[0]
            class_sample_count = id_for_class.shape[0]
            if class_sample_count > 0:
                all_index = numpy.arange(class_sample_count)
                train_index = numpy.random.choice(class_sample_count, train_data_size, replace=False)
                validation_index = numpy.array([index for index in all_index if index not in train_index])
                if validation_size is not None:
                    validation_index_size = validation_index.shape[0]
                    validation_size = min(validation_size, validation_index_size)
                    rand_indices = numpy.random.choice(validation_index_size, validation_size, replace=False)
                    validation_index = validation_index[rand_indices]
                # add elements
                train_set = numpy.vstack([train_set, result_in_clear_area[id_for_class[train_index], :]])
                validation_set = numpy.vstack([validation_set, result_in_clear_area[id_for_class[validation_index], :]])

        return train_set, validation_set
