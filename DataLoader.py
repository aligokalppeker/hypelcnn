from abc import ABC, abstractmethod
from collections import namedtuple

SampleSet = namedtuple('SampleSet', ['validation_targets', 'training_targets', 'test_targets'])
ShadowOperationStruct = namedtuple('ShadowOperationStruct', ['shadow_op', 'shadow_op_creater', 'shadow_op_initializer'])


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, neighborhood, normalize):
        pass

    @abstractmethod
    def load_samples(self, train_data_ratio, test_data_ratio):
        pass

    @abstractmethod
    def load_shadow_map(self, neighborhood, data_set):
        pass

    @abstractmethod
    def get_point_value(self, data_set, point):
        pass

    @abstractmethod
    def get_class_count(self):
        pass

    @abstractmethod
    def get_model_base_dir(self):
        pass

    @abstractmethod
    def get_target_color_list(self):
        pass

    @abstractmethod
    def get_band_measurements(self):
        pass
