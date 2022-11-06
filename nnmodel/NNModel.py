from abc import ABC, abstractmethod


class NNModel(ABC):

    @abstractmethod
    def get_loss_func(self, tensor_output, label):
        pass

    @abstractmethod
    def create_tensor_graph(self, model_input_params, class_count, algorithm_params):
        pass
