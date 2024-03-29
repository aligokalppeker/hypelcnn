from __future__ import division, absolute_import, print_function

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan import gan_loss
from tf_slim import get_variables_to_restore

from gan.wrappers.gan_common import ValidationHook, model_base_name, \
    model_generator_name, adj_shadow_ratio, define_standard_train_ops, create_inference_for_matrix_input, \
    create_input_tensor
from gan.wrappers.wrapper import Wrapper, InferenceWrapper


class GANWrapper(Wrapper):

    def __init__(self, identity_loss_weight, use_identity_loss, swap_inputs,
                 generator_fn, discriminator_fn) -> None:
        super().__init__()
        self._identity_loss_weight = identity_loss_weight
        self._use_identity_loss = use_identity_loss
        self._swap_inputs = swap_inputs
        self._discriminator_fn = discriminator_fn
        self._generator_fn = generator_fn

    def define_model(self, images_x, images_y):
        """Defines a CycleGAN model that maps between images_x and images_y.

        Args:
          images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
          images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.

        Returns:
          A `GANModel` namedtuple.
        """
        with tf.compat.v1.variable_scope(model_base_name, reuse=tf.compat.v1.AUTO_REUSE):
            if self._swap_inputs:
                generator_inputs = images_y
                real_data = images_x
            else:
                generator_inputs = images_x
                real_data = images_y

            return tfgan.gan_model(
                generator_fn=self._generator_fn,
                discriminator_fn=self._discriminator_fn,
                generator_inputs=generator_inputs,
                real_data=real_data)

    def define_loss(self, model):
        with tf.compat.v1.variable_scope(model_base_name, reuse=tf.compat.v1.AUTO_REUSE):
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


class GANInferenceWrapper(InferenceWrapper):
    def __init__(self, fetch_shadows, shadow_generator_fn):
        self._fetch_shadows = fetch_shadows
        self._shadow_generator_fn = shadow_generator_fn

    def construct_inference_graph(self, input_tensor, is_shadow_graph, clip_invalid_values):
        with tf.compat.v1.variable_scope(model_base_name):
            with tf.compat.v1.variable_scope(model_generator_name, reuse=tf.compat.v1.AUTO_REUSE):
                result = create_inference_for_matrix_input(input_tensor, is_shadow_graph, clip_invalid_values,
                                                           self._shadow_generator_fn)
        return result

    def make_inference_graph(self, data_set, is_shadow_graph, clip_invalid_values):
        input_tensor = create_input_tensor(data_set, is_shadow_graph)
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
                              validation_iteration_count, validation_sample_count):
        input_tensor = create_input_tensor(data_set, self._fetch_shadows)
        return ValidationHook(iteration_freq=validation_iteration_count,
                              sample_count=validation_sample_count,
                              log_dir=log_dir,
                              loader=loader, data_set=data_set, neighborhood=neighborhood,
                              shadow_map=shadow_map,
                              shadow_ratio=adj_shadow_ratio(shadow_ratio, self._fetch_shadows),
                              input_tensor=input_tensor,
                              infer_model=self.construct_inference_graph(input_tensor, None, clip_invalid_values=False),
                              fetch_shadows=self._fetch_shadows,
                              name_suffix="deshadowed" if self._fetch_shadows else "shadowed")
