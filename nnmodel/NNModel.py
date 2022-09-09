from abc import ABC, abstractmethod


class NNModel(ABC):
    @abstractmethod
    def get_hyper_param_space(self, trial):
        pass

    @abstractmethod
    def get_default_params(self):
        pass

    @abstractmethod
    def get_loss_func(self, tensor_output, label):
        pass

    @abstractmethod
    def create_tensor_graph(self, model_input_params, class_count, algorithm_params):
        pass
