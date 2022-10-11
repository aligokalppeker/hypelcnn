from functools import partial

import tensorflow as tf
from tensorflow_gan.python import namedtuples
from tensorflow_gan.python.losses import tuple_losses
from tensorflow_gan.python.train import _validate_aux_loss_weight

from gan.wrappers.cut_wrapper import CUTTrainSteps, \
    contrastive_gen_data_x_loss, contrastive_identity_loss
from gan.wrappers.cycle_gan_wrapper import CycleGANInferenceWrapper
from gan.wrappers.dcl_gan_wrapper import dcl_gan_model, dcl_gan_loss, DCLGANWrapper, get_sequential_train_hooks_dclgan
from gan.wrappers.gan_common import model_base_name
from gan.wrappers.wrapper import Wrapper


def dcl_cycle_gan_model(
        # Lambdas defining models.
        generator_fn,
        discriminator_fn,
        feat_discriminator_fn,
        # data from domain x and y.
        image_x,
        image_y,
        # Optional scopes.
        generator_scope="Generator",
        discriminator_scope="Discriminator",
        feat_discriminator_scope="FeatDiscriminator",
        model_x2y_scope='ModelX2Y',
        model_y2x_scope='ModelY2X',
        # Options.
        check_shapes=True):
    dcl_gan_model_ins = dcl_gan_model(generator_fn=generator_fn, discriminator_fn=discriminator_fn,
                                      feat_discriminator_fn=feat_discriminator_fn, image_x=image_x, image_y=image_y,
                                      generator_scope=generator_scope, discriminator_scope=discriminator_scope,
                                      feat_discriminator_scope=feat_discriminator_scope,
                                      model_x2y_scope=model_x2y_scope, model_y2x_scope=model_y2x_scope)

    with tf.compat.v1.variable_scope(dcl_gan_model_ins.model_y2x.generator_scope, reuse=True):
        reconstructed_x = dcl_gan_model_ins.model_y2x.generator_fn(dcl_gan_model_ins.model_x2y.generated_data,
                                                                   create_only_encoder=False)
    with tf.compat.v1.variable_scope(dcl_gan_model_ins.model_x2y.generator_scope, reuse=True):
        reconstructed_y = dcl_gan_model_ins.model_x2y.generator_fn(dcl_gan_model_ins.model_y2x.generated_data,
                                                                   create_only_encoder=False)

    return namedtuples.CycleGANModel(dcl_gan_model_ins.model_x2y, dcl_gan_model_ins.model_y2x,
                                     reconstructed_x, reconstructed_y)


def dcl_cycle_gan_loss(
        model,
        # Loss functions.
        generator_loss_fn=tuple_losses.wasserstein_generator_loss,
        discriminator_loss_fn=tuple_losses.wasserstein_discriminator_loss,
        gen_discriminator_loss_fn=tuple_losses.wasserstein_discriminator_loss,
        identity_discriminator_loss_fn=tuple_losses.wasserstein_discriminator_loss,
        cycle_consistency_loss_fn=tuple_losses.cycle_consistency_loss,
        cycle_consistency_loss_weight=10.0,
        # Auxiliary losses.
        gradient_penalty_weight=None,
        gradient_penalty_epsilon=1e-10,
        gradient_penalty_target=1.0,
        gradient_penalty_one_sided=False,
        mutual_information_penalty_weight=None,
        aux_cond_generator_weight=None,
        aux_cond_discriminator_weight=None,
        # Options.
        nce_loss_weight=10.0,
        nce_identity_loss_weight=2,
        reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=True):
    """Returns losses necessary to train generator and discriminator.

    Args:
      cycle_consistency_loss_weight:
      cycle_consistency_loss_fn:
      identity_discriminator_loss_fn:
      nce_identity_loss_weight:
      nce_loss_weight: NCE loss weight
      gen_discriminator_loss_fn: Generator discriminator loss function
      model: A GANModel tuple.
      generator_loss_fn: The loss function on the generator. Takes a GANModel
        tuple. If it also takes `reduction` or `add_summaries`, it will be
        passed those values as well. All TF-GAN loss functions have these
        arguments.
      discriminator_loss_fn: The loss function on the discriminator. Takes a
        GANModel tuple. If it also takes `reduction` or `add_summaries`, it will
        be passed those values as well. All TF-GAN loss functions have these
        arguments.
      gradient_penalty_weight: If not `None`, must be a non-negative Python number
        or Tensor indicating how much to weight the gradient penalty. See
        https://arxiv.org/pdf/1704.00028.pdf for more details.
      gradient_penalty_epsilon: If `gradient_penalty_weight` is not None, the
        small positive value used by the gradient penalty function for numerical
        stability. Note some applications will need to increase this value to
        avoid NaNs.
      gradient_penalty_target: If `gradient_penalty_weight` is not None, a Python
        number or `Tensor` indicating the target value of gradient norm. See the
        CIFAR10 section of https://arxiv.org/abs/1710.10196. Defaults to 1.0.
      gradient_penalty_one_sided: If `True`, penalty proposed in
        https://arxiv.org/abs/1709.08894 is used. Defaults to `False`.
      mutual_information_penalty_weight: If not `None`, must be a non-negative
        Python number or Tensor indicating how much to weight the mutual
        information penalty. See https://arxiv.org/abs/1606.03657 for more
        details.
      aux_cond_generator_weight: If not None: add a classification loss as in
        https://arxiv.org/abs/1610.09585
      aux_cond_discriminator_weight: If not None: add a classification loss as in
        https://arxiv.org/abs/1610.09585
      tensor_pool_fn: A function that takes (generated_data, generator_inputs),
        stores them in an internal pool and returns previous stored
        (generated_data, generator_inputs). For example
        `tfgan.features.tensor_pool`. Defaults to None (not using tensor pool).
      reduction: A `tf.losses.Reduction` to apply to loss, if the loss takes an
        argument called `reduction`. Otherwise, this is ignored.
      add_summaries: Whether to add summaries for the losses.

    Returns:
      A GANLoss 2-tuple of (generator_loss, discriminator_loss). Includes
      regularization losses.

    Raises:
      ValueError: If any of the auxiliary loss weights is provided and negative.
      ValueError: If `mutual_information_penalty_weight` is provided, but the
        `model` isn't an `InfoGANModel`.
    """

    dcl_gan_loss_inst = dcl_gan_loss(model=model, generator_loss_fn=generator_loss_fn,
                                     discriminator_loss_fn=discriminator_loss_fn,
                                     gen_discriminator_loss_fn=gen_discriminator_loss_fn,
                                     identity_discriminator_loss_fn=identity_discriminator_loss_fn,
                                     gradient_penalty_weight=gradient_penalty_weight,
                                     gradient_penalty_epsilon=gradient_penalty_epsilon,
                                     gradient_penalty_target=gradient_penalty_target,
                                     gradient_penalty_one_sided=gradient_penalty_one_sided,
                                     mutual_information_penalty_weight=mutual_information_penalty_weight,
                                     aux_cond_generator_weight=aux_cond_generator_weight,
                                     aux_cond_discriminator_weight=aux_cond_discriminator_weight,
                                     nce_loss_weight=nce_loss_weight,
                                     nce_identity_loss_weight=nce_identity_loss_weight, reduction=reduction,
                                     add_summaries=add_summaries)

    # Defines cycle consistency loss.
    cycle_consistency_loss = cycle_consistency_loss_fn(
        model, add_summaries=add_summaries)
    cycle_consistency_loss_weight = _validate_aux_loss_weight(
        cycle_consistency_loss_weight, 'cycle_consistency_loss_weight')
    aux_loss = cycle_consistency_loss_weight * cycle_consistency_loss

    dcl_gan_loss_inst.loss_x2y._replace(generator_loss=dcl_gan_loss_inst.loss_x2y.generator_loss + aux_loss)
    dcl_gan_loss_inst.loss_y2x._replace(generator_loss=dcl_gan_loss_inst.loss_y2x.generator_loss + aux_loss)

    return namedtuples.CycleGANLoss(dcl_gan_loss_inst.loss_x2y, dcl_gan_loss_inst.loss_y2x)


class DCLCycleGANWrapper(Wrapper):

    def __init__(self, nce_loss_weight, identity_loss_weight, cycle_consistency_loss_weight,
                 use_identity_loss, tau, batch_size,
                 generator_fn, discriminator_fn, feat_discriminator_fn) -> None:
        super().__init__()
        self._cycle_consistency_loss_weight = cycle_consistency_loss_weight
        self._nce_loss_weight = nce_loss_weight
        self._identity_loss_weight = 0.0 if not use_identity_loss else identity_loss_weight
        self._tau = tau
        self._batch_size = batch_size
        self._generator_fn = generator_fn
        self._discriminator_fn = discriminator_fn
        self._feat_discriminator_fn = feat_discriminator_fn

    def define_model(self, images_x, images_y):
        """Defines a CycleGAN model that maps between images_x and images_y.

        Args:
          images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
          images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.

        Returns:
          A `DCLGANModel` namedtuple.
        """
        with tf.compat.v1.variable_scope(model_base_name, reuse=tf.compat.v1.AUTO_REUSE):
            return dcl_cycle_gan_model(
                generator_fn=self._generator_fn,
                discriminator_fn=self._discriminator_fn,
                feat_discriminator_fn=self._feat_discriminator_fn,
                image_x=images_x,
                image_y=images_y)

    def define_loss(self, model):
        with tf.compat.v1.variable_scope(model_base_name, reuse=tf.compat.v1.AUTO_REUSE):
            # Define CycleGAN loss.
            return dcl_cycle_gan_loss(model,
                                      generator_loss_fn=tuple_losses.least_squares_generator_loss,
                                      discriminator_loss_fn=tuple_losses.least_squares_discriminator_loss,
                                      gen_discriminator_loss_fn=partial(contrastive_gen_data_x_loss, tau=self._tau,
                                                                        batch_size=self._batch_size),
                                      identity_discriminator_loss_fn=partial(contrastive_identity_loss, tau=self._tau,
                                                                             batch_size=self._batch_size),
                                      nce_loss_weight=self._nce_loss_weight,
                                      nce_identity_loss_weight=self._identity_loss_weight,
                                      cycle_consistency_loss_weight=self._cycle_consistency_loss_weight)

    def define_train_ops(self, model, loss, max_number_of_steps, **kwargs):
        return DCLGANWrapper.base_trainops_method(model, loss, max_number_of_steps, kwargs)

    def get_train_hooks_fn(self):
        return get_sequential_train_hooks_dclgan(CUTTrainSteps(1, 1, 1))


class DCLCycleGANInferenceWrapper(CycleGANInferenceWrapper):
    def __init__(self, shadow_generator_fn) -> None:
        super().__init__(shadow_generator_fn)
