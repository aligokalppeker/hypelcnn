from abc import ABC, abstractmethod


class Wrapper(ABC):
    @abstractmethod
    def define_model(self, images_x, images_y):
        pass

    @abstractmethod
    def define_loss(self, model):
        pass

    @abstractmethod
    def define_train_ops(self, model, loss, max_number_of_steps, **kwargs):
        pass

    @abstractmethod
    def get_train_hooks_fn(self):
        pass


class InferenceWrapper:
    @abstractmethod
    def construct_inference_graph(self, input_tensor, is_shadow_graph, clip_invalid_values):
        pass

    @abstractmethod
    def make_inference_graph(self, data_set, is_shadow_graph, clip_invalid_values):
        pass

    @abstractmethod
    def create_generator_restorer(self):
        pass

    @abstractmethod
    def create_inference_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                              validation_iteration_count, validation_sample_count):
        pass
