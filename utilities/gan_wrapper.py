from __future__ import division, absolute_import, print_function

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan import gan_loss

from shadow_data_generator import _shadowdata_generator_model, _shadowdata_discriminator_model
from gan_common import ValidationHook


class GANWrapper:

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

        # Add summaries for generated images.
        # tfgan.eval.add_cyclegan_image_summaries(gan_model)

        return gan_model

    def define_loss(self, model):
        # Define CycleGAN loss.
        loss = gan_loss(
            model,
            # generator_loss_fn=wasserstein_generator_loss,
            # discriminator_loss_fn=wasserstein_discriminator_loss,
            tensor_pool_fn=tfgan.features.tensor_pool)
        return loss

    def create_validation_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                               validation_iteration_count, validation_sample_count):
        element_size = loader.get_data_shape(data_set)
        element_size = [1, element_size[0], element_size[1], element_size[2] - 1]

        x_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='x')
        y_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name='y')
        model_for_validation = self.define_model(x_input_tensor, y_input_tensor)
        shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                                  sample_count=validation_sample_count,
                                                  log_dir=log_dir,
                                                  loader=loader, data_set=data_set, neighborhood=neighborhood,
                                                  shadow_map=shadow_map,
                                                  shadow_ratio=shadow_ratio,
                                                  input_tensor=x_input_tensor,
                                                  model=model_for_validation.generated_data,
                                                  fetch_shadows=False, name_suffix="shadowed")
        return shadowed_validation_hook
