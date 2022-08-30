from __future__ import division, absolute_import, print_function

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan import gan_loss
from tf_slim import get_variables_to_restore

from gan.shadow_data_models import _shadowdata_generator_model, _shadowdata_discriminator_model
from gan.wrappers.gan_common import ValidationHook, input_x_tensor_name, input_y_tensor_name, model_base_name, \
    model_generator_name, adj_shadow_ratio, define_standard_train_ops, create_inference_for_matrix_input, \
    define_val_model
from gan.wrappers.wrapper import Wrapper, InferenceWrapper


class GANWrapper(Wrapper):

    def __init__(self, identity_loss_weight, use_identity_loss, swap_inputs) -> None:
        super().__init__()
        self._identity_loss_weight = identity_loss_weight
        self._use_identity_loss = use_identity_loss
        self._swap_inputs = swap_inputs

    def define_model(self, images_x, images_y):
        """Defines a CycleGAN model that maps between images_x and images_y.

        Args:
          images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
          images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.
          use_identity_loss: Whether to use identity loss or not

        Returns:
          A `CycleGANModel` namedtuple.
        """
        if self._swap_inputs:
            generator_inputs = images_y
            real_data = images_x
        else:
            generator_inputs = images_x
            real_data = images_y

        gan_model = tfgan.gan_model(
            generator_fn=_shadowdata_generator_model,
            discriminator_fn=_shadowdata_discriminator_model,
            generator_inputs=generator_inputs,
            real_data=real_data)

        return gan_model

    def define_loss(self, model):
        # Define CycleGAN loss.
        loss = gan_loss(
            model,
            # generator_loss_fn=wasserstein_generator_loss,
            # discriminator_loss_fn=wasserstein_discriminator_loss,
            tensor_pool_fn=tfgan.features.tensor_pool)
        return loss

    def define_train_ops(self, model, loss, max_number_of_steps, **kwargs):
        return define_standard_train_ops(model, loss,
                                         max_number_of_steps=max_number_of_steps,
                                         generator_lr=kwargs["generator_lr"],
                                         discriminator_lr=kwargs["discriminator_lr"])

    def get_train_hooks_fn(self):
        return tfgan.get_sequential_train_hooks(tfgan.GANTrainSteps(1, 1))

    @staticmethod
    def create_validation_hook_base(wrapper, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                                    validation_iteration_count, validation_sample_count, swap_inputs):
        model_for_validation, x_input_tensor, y_input_tensor = define_val_model(wrapper, data_set)
        shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                                  sample_count=validation_sample_count,
                                                  log_dir=log_dir,
                                                  loader=loader, data_set=data_set, neighborhood=neighborhood,
                                                  shadow_map=shadow_map,
                                                  shadow_ratio=adj_shadow_ratio(shadow_ratio, swap_inputs),
                                                  input_tensor=y_input_tensor if swap_inputs else x_input_tensor,
                                                  infer_model=model_for_validation.generated_data,
                                                  fetch_shadows=False,
                                                  name_suffix="deshadowed" if swap_inputs else "shadowed")
        return shadowed_validation_hook

    def create_validation_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                               validation_iteration_count, validation_sample_count):
        return GANWrapper.create_validation_hook_base(self, data_set, loader, log_dir, neighborhood, shadow_map,
                                                      shadow_ratio, validation_iteration_count, validation_sample_count,
                                                      self._swap_inputs)


class GANInferenceWrapper(InferenceWrapper):
    def __init__(self, fetch_shadows):
        self.fetch_shadows = fetch_shadows

    def construct_inference_graph(self, input_tensor, is_shadow_graph, clip_invalid_values):
        with tf.compat.v1.variable_scope(model_base_name):
            with tf.compat.v1.variable_scope(model_generator_name, reuse=tf.compat.v1.AUTO_REUSE):
                result = create_inference_for_matrix_input(input_tensor, is_shadow_graph, clip_invalid_values)
        return result

    @staticmethod
    def __create_input_tensor(data_set, is_shadow_graph):
        element_size = data_set.get_data_shape()
        element_size = [None, element_size[0], element_size[1], data_set.get_casi_band_count()]
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=element_size,
                                                name=input_x_tensor_name if is_shadow_graph else input_y_tensor_name)
        return input_tensor

    def make_inference_graph(self, data_set, is_shadow_graph, clip_invalid_values):
        input_tensor = self.__create_input_tensor(data_set, is_shadow_graph)
        generated = self.construct_inference_graph(input_tensor, is_shadow_graph, clip_invalid_values)
        return input_tensor, generated

    def create_generator_restorer(self):
        # Restore all the variables that were saved in the checkpoint.
        gan_restorer = tf.compat.v1.train.Saver(
            var_list=get_variables_to_restore(include=[model_base_name]),
            name="GeneratorRestoreHandler"
        )
        return gan_restorer

    def create_inference_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                              validation_sample_count):
        input_tensor = self.__create_input_tensor(data_set, self.fetch_shadows)
        return ValidationHook(iteration_freq=0,
                              sample_count=validation_sample_count,
                              log_dir=log_dir,
                              loader=loader, data_set=data_set, neighborhood=neighborhood,
                              shadow_map=shadow_map,
                              shadow_ratio=adj_shadow_ratio(shadow_ratio, self.fetch_shadows),
                              input_tensor=input_tensor,
                              infer_model=self.construct_inference_graph(input_tensor, None, clip_invalid_values=False),
                              fetch_shadows=self.fetch_shadows,
                              name_suffix="")
