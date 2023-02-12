from abc import ABC, abstractmethod
from enum import Enum


class SampleSet:
    def __init__(self, validation_targets, training_targets, test_targets) -> None:
        super().__init__()
        self.validation_targets = validation_targets
        self.training_targets = training_targets
        self.test_targets = test_targets


class LoadingMode(Enum):
    ORIGINAL = ""
    SHADOWED = "shadowed"
    DESHADOWED = "deshadowed"
    MIXED = "mixed"


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
    def get_class_count(self):
        pass

    @abstractmethod
    def get_model_base_dir(self):
        pass

    @abstractmethod
    def get_samples_color_list(self):
        pass

    @abstractmethod
    def get_band_measurements(self):
        pass
