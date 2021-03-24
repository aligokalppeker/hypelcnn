from abc import ABC, abstractmethod


class DataImporter(ABC):
    @abstractmethod
    def read_data_set(self, loader_name, path, train_data_ratio, test_data_ratio, neighborhood, normalize):
        pass

    @abstractmethod
    def convert_data_to_tensor(self, test_data_with_labels, training_data_with_labels, validation_data_with_labels,
                               class_range):
        pass

    @abstractmethod
    def perform_tensor_initialize(self, session, tensor, nn_params):
        pass

    @abstractmethod
    def requires_separate_validation_branch(self):
        pass

    @abstractmethod
    def create_all_scene_data(self, scene_shape, data_with_labels_to_copy):
        pass
