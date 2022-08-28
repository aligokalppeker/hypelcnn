from __future__ import division, absolute_import, print_function

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan import gan_loss
from tensorflow_gan.python import namedtuples
from tensorflow_gan.python.losses import tuple_losses
from tensorflow_gan.python.train import _validate_aux_loss_weight
from tf_slim import get_variables_to_restore

from gan.shadow_data_models import _shadowdata_generator_model, _shadowdata_discriminator_model
from gan.wrappers.gan_common import PeerValidationHook, ValidationHook, input_x_tensor_name, input_y_tensor_name, \
    model_base_name, \
    model_generator_name, define_standard_train_ops, create_inference_for_matrix_input

model_forward_generator_name = "ModelX2Y"
model_backward_generator_name = "ModelY2X"


def create_base_validation_hook(data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                                validation_iteration_count, validation_sample_count,
                                model_forward, model_backward,
                                x_input_tensor, y_input_tensor):
    shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                              sample_count=validation_sample_count,
                                              log_dir=log_dir,
                                              loader=loader, data_set=data_set, neighborhood=neighborhood,
                                              shadow_map=shadow_map,
                                              shadow_ratio=shadow_ratio,
                                              input_tensor=x_input_tensor,
                                              infer_model=model_forward,
                                              fetch_shadows=False, name_suffix="shadowed")
    de_shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                                 sample_count=validation_sample_count,
                                                 log_dir=log_dir,
                                                 loader=loader, data_set=data_set, neighborhood=neighborhood,
                                                 shadow_map=shadow_map,
                                                 shadow_ratio=1. / shadow_ratio,
                                                 input_tensor=y_input_tensor,
                                                 infer_model=model_backward,
                                                 fetch_shadows=True, name_suffix="deshadowed")
    peer_validation_hook = PeerValidationHook(shadowed_validation_hook, de_shadowed_validation_hook)
    return peer_validation_hook


class CycleGANWrapper:

    def __init__(self, cycle_consistency_loss_weight, identity_loss_weight, use_identity_loss) -> None:
        super().__init__()
        self._cycle_consistency_loss_weight = cycle_consistency_loss_weight
        self._identity_loss_weight = identity_loss_weight
        self._use_identity_loss = use_identity_loss

    def define_model(self, images_x, images_y):
        """Defines a CycleGAN model that maps between images_x and images_y.

        Args:
          images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
          images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.
          use_identity_loss: Whether to use identity loss or not

        Returns:
          A `CycleGANModel` namedtuple.
        """
        if self._use_identity_loss:
            cyclegan_model = cyclegan_model_with_identity(
                generator_fn=_shadowdata_generator_model,
                discriminator_fn=_shadowdata_discriminator_model,
                data_x=images_x,
                data_y=images_y)
        else:
            cyclegan_model = tfgan.cyclegan_model(
                generator_fn=_shadowdata_generator_model,
                discriminator_fn=_shadowdata_discriminator_model,
                data_x=images_x,
                data_y=images_y)

        # Add summaries for generated images.
        # tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

        return cyclegan_model

    def define_loss(self, model):
        if self._use_identity_loss:
            cyclegan_loss = cyclegan_loss_with_identity(
                model,
                # generator_loss_fn=wasserstein_generator_loss,
                # discriminator_loss_fn=wasserstein_discriminator_loss,
                cycle_consistency_loss_weight=self._cycle_consistency_loss_weight,
                identity_loss_weight=self._identity_loss_weight,
                tensor_pool_fn=tfgan.features.tensor_pool)
        else:
            # Define CycleGAN loss.
            cyclegan_loss = tfgan.cyclegan_loss(
                model,
                # generator_loss_fn=wasserstein_generator_loss,
                # discriminator_loss_fn=wasserstein_discriminator_loss,
                cycle_consistency_loss_weight=self._cycle_consistency_loss_weight,
                tensor_pool_fn=tfgan.features.tensor_pool)
        return cyclegan_loss

    def define_train_ops(self, model, loss, max_number_of_steps, **kwargs):
        return define_standard_train_ops(model, loss,
                                         max_number_of_steps=max_number_of_steps,
                                         generator_lr=kwargs["generator_lr"],
                                         discriminator_lr=kwargs["discriminator_lr"])

    def get_train_hooks_fn(self):
        return tfgan.get_sequential_train_hooks(tfgan.GANTrainSteps(1, 1))

    def create_validation_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                               validation_iteration_count, validation_sample_count):
        element_size = data_set.get_data_shape()
        element_size = [None, element_size[0], element_size[1], data_set.get_casi_band_count()]

        x_input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=element_size, name=input_x_tensor_name)
        y_input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=element_size, name=input_y_tensor_name)
        cyclegan_model_for_validation = self.define_model(x_input_tensor, y_input_tensor)
        return create_base_validation_hook(data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                                           validation_iteration_count, validation_sample_count,
                                           cyclegan_model_for_validation.model_x2y.generated_data,
                                           cyclegan_model_for_validation.model_y2x.generated_data,
                                           x_input_tensor, y_input_tensor)


class CycleGANInferenceWrapper:
    def construct_inference_graph(self, input_tensor, is_shadow_graph, clip_invalid_values=False):
        model_name = model_forward_generator_name if is_shadow_graph else model_backward_generator_name
        with tf.compat.v1.variable_scope(model_base_name):
            with tf.compat.v1.variable_scope(model_name):
                with tf.compat.v1.variable_scope(model_generator_name, reuse=tf.compat.v1.AUTO_REUSE):
                    result = create_inference_for_matrix_input(input_tensor, is_shadow_graph, clip_invalid_values)

        return result

    def make_inference_graph(self, data_set, is_shadow_graph, clip_invalid_values):
        element_size = data_set.get_data_shape()
        element_size = [None, element_size[0], element_size[1], data_set.get_casi_band_count()]
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=element_size, name=input_x_tensor_name)
        generated = self.construct_inference_graph(input_tensor, is_shadow_graph, clip_invalid_values)
        return input_tensor, generated

    def create_generator_restorer(self):
        # Restore all the variables that were saved in the checkpoint.
        cyclegan_restorer = tf.compat.v1.train.Saver(
            var_list=get_variables_to_restore(include=[model_base_name + "/" + model_forward_generator_name]) +
                     get_variables_to_restore(include=[model_base_name + "/" + model_backward_generator_name]),
            name="GeneratorRestoreHandler"
        )
        return cyclegan_restorer

    def create_inference_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                              validation_sample_count):
        element_size = data_set.get_data_shape()
        element_size = [None, element_size[0], element_size[1], data_set.get_casi_band_count()]

        x_input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=element_size, name='x')
        y_input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=element_size, name='y')
        return create_base_validation_hook(data_set=data_set,
                                           loader=loader,
                                           log_dir=log_dir,
                                           neighborhood=neighborhood,
                                           shadow_map=shadow_map,
                                           shadow_ratio=shadow_ratio,
                                           validation_iteration_count=0,
                                           validation_sample_count=validation_sample_count,
                                           model_forward=self.construct_inference_graph(x_input_tensor, True),
                                           model_backward=self.construct_inference_graph(y_input_tensor, False),
                                           x_input_tensor=x_input_tensor, y_input_tensor=y_input_tensor)


class CycleGANModelWithIdentity(tfgan.CycleGANModel):
    """An CycleGANModel contains all the pieces needed for CycleGAN training and additional outputs for calculating identity loss.

    The model `model_x2y` generator F maps data set X to Y, while the model
    `model_y2x` generator G maps data set Y to X.

    See https://arxiv.org/abs/1703.10593 for more details.

    Args:
      model_x2y: A `GANModel` namedtuple whose generator maps data set X to Y.
      model_y2x: A `GANModel` namedtuple whose generator maps data set Y to X.
      reconstructed_x: A `Tensor` of reconstructed data X which is G(F(X)).
      reconstructed_y: A `Tensor` of reconstructed data Y which is F(G(Y)).
      identity_x: A `Tensor` of data X processed by generator function F which is F(X).
      identity_y: A `Tensor` of data Y processed by generator function G which is G(Y).
    """
    __slots__ = ()
    _fields = tfgan.CycleGANModel._fields + ("identity_x", "identity_y",)


def cyclegan_loss_with_identity(
        model,
        # Loss functions.
        generator_loss_fn=tuple_losses.least_squares_generator_loss,
        discriminator_loss_fn=tuple_losses.least_squares_discriminator_loss,
        # Auxiliary losses.
        cycle_consistency_loss_fn=tuple_losses.cycle_consistency_loss,
        cycle_consistency_loss_weight=10.0,
        identity_loss_weight=0.5,
        # Options
        **kwargs):
    """Returns the losses for a `CycleGANModel`.

    See https://arxiv.org/abs/1703.10593 for more details.

    Args:
      model: A `CycleGANModel` namedtuple.
      generator_loss_fn: The loss function on the generator. Takes a `GANModel`
        named tuple.
      discriminator_loss_fn: The loss function on the discriminator. Takes a
        `GANModel` namedtuple.
      cycle_consistency_loss_fn: The cycle consistency loss function. Takes a
        `CycleGANModel` namedtuple.
      cycle_consistency_loss_weight: A non-negative Python number or a scalar
        `Tensor` indicating how much to weigh the cycle consistency loss.
      identity_loss_weight: A non-negative Python number or a scalar
        `Tensor` indicating how much to weigh the identity loss.
      **kwargs: Keyword args to pass directly to `gan_loss` to construct the loss
        for each partial model of `model`.

    Returns:
      A `CycleGANLoss` namedtuple.

    Raises:
      ValueError: If `model` is not a `CycleGANModel` namedtuple.
    """
    # Sanity checks.
    if not isinstance(model, CycleGANModelWithIdentity):
        raise ValueError(
            f"`model` must be a `CycleGANModelWithIdentity`. Instead, was {type(model)}.")

    identity_loss_x, identity_loss_y = identity_loss(model, kwargs)

    # Defines cycle consistency loss.
    cycle_consistency_loss = cycle_consistency_loss_fn(
        model, add_summaries=kwargs.get("add_summaries", True))
    cycle_consistency_loss_weight = _validate_aux_loss_weight(
        cycle_consistency_loss_weight, "cycle_consistency_loss_weight")

    aux_loss = (cycle_consistency_loss_weight * cycle_consistency_loss) + (
            identity_loss_weight * (identity_loss_x + identity_loss_y))

    # Defines losses for each partial model.
    def _partial_loss(partial_model):
        partial_loss = gan_loss(
            partial_model,
            generator_loss_fn=generator_loss_fn,
            discriminator_loss_fn=discriminator_loss_fn,
            **kwargs)
        return partial_loss._replace(generator_loss=partial_loss.generator_loss + aux_loss)

    with tf.compat.v1.name_scope("cyclegan_loss_x2y"):
        loss_x2y = _partial_loss(model.model_x2y)
    with tf.compat.v1.name_scope("cyclegan_loss_y2x"):
        loss_y2x = _partial_loss(model.model_y2x)

    return namedtuples.CycleGANLoss(loss_x2y, loss_y2x)


def cyclegan_model_with_identity(
        # Lambdas defining models.
        generator_fn,
        discriminator_fn,
        # data X and Y.
        data_x,
        data_y,
        # Optional scopes.
        generator_scope="Generator",
        discriminator_scope="Discriminator",
        model_x2y_scope="ModelX2Y",
        model_y2x_scope="ModelY2X",
        # Options.
        check_shapes=True):
    """Returns a CycleGAN model outputs and variables.

    See https://arxiv.org/abs/1703.10593 for more details.

    Args:
      generator_fn: A python lambda that takes `data_x` or `data_y` as inputs and
        returns the outputs of the GAN generator.
      discriminator_fn: A python lambda that takes `real_data`/`generated data`
        and `generator_inputs`. Outputs a Tensor in the range [-inf, inf].
      data_x: A `Tensor` of dataset X. Must be the same shape as `data_y`.
      data_y: A `Tensor` of dataset Y. Must be the same shape as `data_x`.
      generator_scope: Optional generator variable scope. Useful if you want to
        reuse a subgraph that has already been created. Defaults to 'Generator'.
      discriminator_scope: Optional discriminator variable scope. Useful if you
        want to reuse a subgraph that has already been created. Defaults to
        'Discriminator'.
      model_x2y_scope: Optional variable scope for model x2y variables. Defaults
        to 'ModelX2Y'.
      model_y2x_scope: Optional variable scope for model y2x variables. Defaults
        to 'ModelY2X'.
      check_shapes: If `True`, check that generator produces Tensors that are the
        same shape as `data_x` (`data_y`). Otherwise, skip this check.

    Returns:
      A `CycleGANModel` namedtuple.

    Raises:
      ValueError: If `check_shapes` is True and `data_x` or the generator output
        does not have the same shape as `data_y`.
      ValueError: If TF is executing eagerly.
    """
    original_model = tfgan.cyclegan_model(generator_fn=generator_fn, discriminator_fn=discriminator_fn, data_x=data_x,
                                          data_y=data_y, generator_scope=generator_scope,
                                          discriminator_scope=discriminator_scope, model_x2y_scope=model_x2y_scope,
                                          model_y2x_scope=model_y2x_scope, check_shapes=check_shapes)

    with tf.compat.v1.variable_scope(original_model.model_x2y.generator_scope, reuse=True):
        identity_x = original_model.model_x2y.generator_fn(data_x)
    with tf.compat.v1.variable_scope(original_model.model_y2x.generator_scope, reuse=True):
        identity_y = original_model.model_y2x.generator_fn(data_y)

    model_w_identity = CycleGANModelWithIdentity(model_x2y=original_model.model_x2y,
                                                 model_y2x=original_model.model_y2x,
                                                 reconstructed_x=original_model.reconstructed_x,
                                                 reconstructed_y=original_model.reconstructed_y)
    model_w_identity.identity_x = identity_x
    model_w_identity.identity_y = identity_y

    return model_w_identity


def identity_loss(model, kwargs):
    # Defines identity loss
    identity_loss_x = tf.compat.v1.losses.absolute_difference(model.model_x2y.generator_inputs,
                                                              model.identity_x)
    identity_loss_y = tf.compat.v1.losses.absolute_difference(model.model_y2x.generator_inputs,
                                                              model.identity_y)
    if kwargs.get("add_summaries", True):
        tf.compat.v1.summary.scalar("identity_loss_x", identity_loss_x)
        tf.compat.v1.summary.scalar("identity_loss_y", identity_loss_y)

    return identity_loss_x, identity_loss_y
