import collections
from functools import partial

import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow_gan.python.losses import tuple_losses

from gan.shadow_data_models import _shadowdata_generator_model, _shadowdata_discriminator_model, \
    _shadowdata_feature_discriminator_model
from gan.wrappers.cut_wrapper import cut_model, cut_loss, cut_train_ops, get_sequential_train_hooks_cut, CUTTrainSteps, \
    contrastive_gen_data_x_loss, contrastive_identity_loss
from gan.wrappers.cycle_gan_wrapper import create_base_validation_hook, CycleGANInferenceWrapper
from gan.wrappers.gan_common import _get_lr, define_val_model
from gan.wrappers.wrapper import Wrapper


class DCLGANModel(
    collections.namedtuple("DCLGANModel", (
            "model_x2y",
            "model_y2x",
    ))):
    """A DCLGANModel contains all the pieces needed for training.

    Args:
      model_x2y: CUT model for shadowing transformation.
      model_y2x: CUT model for de-shadowing transformation.
    """


def dcl_gan_model(
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
    with tf.compat.v1.variable_scope(model_x2y_scope):
        cut_model_x2y = cut_model(
            generator_fn=generator_fn,
            discriminator_fn=discriminator_fn,
            feat_discriminator_fn=feat_discriminator_fn,
            generator_inputs=image_x,
            real_data=image_y
            # ,generator_scope="Generator_x2y",
            # discriminator_scope="Discriminator_x2y",
            # feat_discriminator_scope="FeatDiscriminator_x2y"
        )

    with tf.compat.v1.variable_scope(model_y2x_scope):
        cut_model_y2x = cut_model(
            generator_fn=generator_fn,
            discriminator_fn=discriminator_fn,
            feat_discriminator_fn=feat_discriminator_fn,
            generator_inputs=image_y,
            real_data=image_x
            # ,generator_scope="Generator_y2x",
            # discriminator_scope="Discriminator_y2x",
            # feat_discriminator_scope="FeatDiscriminator_y2x"
        )

    return DCLGANModel(model_x2y=cut_model_x2y, model_y2x=cut_model_y2x)


class DCLGANLoss(collections.namedtuple("DCLGANLoss", (
        "loss_x2y",
        "loss_y2x"))):
    """DCLGANLoss contains the CUT loss for x2y and y2x.

    Args:
      loss_x2y: A tensor for the generator loss.
      loss_y2x: A tensor for the discriminator loss.
    """


def dcl_gan_loss(
        model,
        # Loss functions.
        generator_loss_fn=tuple_losses.wasserstein_generator_loss,
        discriminator_loss_fn=tuple_losses.wasserstein_discriminator_loss,
        gen_discriminator_loss_fn=tuple_losses.wasserstein_discriminator_loss,
        identity_discriminator_loss_fn=tuple_losses.wasserstein_discriminator_loss,
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
    loss_x2y = cut_loss(model.model_x2y,
                        generator_loss_fn=generator_loss_fn,
                        discriminator_loss_fn=discriminator_loss_fn,
                        gen_discriminator_loss_fn=gen_discriminator_loss_fn,
                        identity_discriminator_loss_fn=identity_discriminator_loss_fn,
                        nce_loss_weight=nce_loss_weight,
                        nce_identity_loss_weight=nce_identity_loss_weight,
                        gradient_penalty_weight=gradient_penalty_weight,
                        gradient_penalty_epsilon=gradient_penalty_epsilon,
                        gradient_penalty_target=gradient_penalty_target,
                        gradient_penalty_one_sided=gradient_penalty_one_sided,
                        mutual_information_penalty_weight=mutual_information_penalty_weight,
                        aux_cond_generator_weight=aux_cond_generator_weight,
                        aux_cond_discriminator_weight=aux_cond_discriminator_weight,
                        reduction=reduction,
                        add_summaries=add_summaries)

    loss_y2x = cut_loss(model.model_y2x,
                        generator_loss_fn=generator_loss_fn,
                        discriminator_loss_fn=discriminator_loss_fn,
                        gen_discriminator_loss_fn=gen_discriminator_loss_fn,
                        identity_discriminator_loss_fn=identity_discriminator_loss_fn,
                        nce_loss_weight=nce_loss_weight,
                        nce_identity_loss_weight=nce_identity_loss_weight,
                        gradient_penalty_weight=gradient_penalty_weight,
                        gradient_penalty_epsilon=gradient_penalty_epsilon,
                        gradient_penalty_target=gradient_penalty_target,
                        gradient_penalty_one_sided=gradient_penalty_one_sided,
                        mutual_information_penalty_weight=mutual_information_penalty_weight,
                        aux_cond_generator_weight=aux_cond_generator_weight,
                        aux_cond_discriminator_weight=aux_cond_discriminator_weight,
                        reduction=reduction,
                        add_summaries=add_summaries)

    loss_x2y._replace(generator_loss=loss_x2y.generator_loss + loss_y2x.generator_loss)
    loss_y2x._replace(generator_loss=loss_y2x.generator_loss + loss_x2y.generator_loss)

    return DCLGANLoss(loss_x2y=loss_x2y, loss_y2x=loss_y2x)


class DCLGANTrainOps(
    collections.namedtuple("DCLGANTrainOps", (
            "x2y_ops",
            "y2x_ops",
            "global_step_inc_op",
            "train_hooks",))):
    """CUTTrainOps contains the training ops.

    Args:
      generator_train_op: Op that performs a generator update step.
      discriminator_train_op: Op that performs a discriminator update step.
      global_step_inc_op: Op that increments the shared global step.
      train_hooks: a list or tuple containing hooks related to training that need
        to be populated when training ops are instantiated. Used primarily for
        sync hooks.
    """


def get_sequential_train_hooks_dclgan(train_steps):
    """Returns a hooks function for sequential GAN training.

    Args:
      train_steps: A `GANTrainSteps` tuple that determines how many generator
        and discriminator training steps to take.

    Returns:
      A function that takes a GANTrainOps tuple and returns a list of hooks.
    """

    def get_hooks(train_ops):
        x2y_ops_seq_list = get_sequential_train_hooks_cut(train_steps=train_steps)(train_ops.x2y_ops)
        y2x_ops_seq_list = get_sequential_train_hooks_cut(train_steps=train_steps)(train_ops.y2x_ops)
        return x2y_ops_seq_list + y2x_ops_seq_list

    return get_hooks


class DCLGANWrapper(Wrapper):

    def __init__(self, nce_loss_weight, identity_loss_weight, use_identity_loss, tau, batch_size) -> None:
        super().__init__()
        self._nce_loss_weight = nce_loss_weight
        self._identity_loss_weight = 0.0 if not use_identity_loss else identity_loss_weight
        self._tau = tau
        self._batch_size = batch_size

    def define_model(self, images_x, images_y):
        """Defines a CycleGAN model that maps between images_x and images_y.

        Args:
          images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
          images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.

        Returns:
          A `DCLGANModel` namedtuple.
        """

        return dcl_gan_model(
            generator_fn=_shadowdata_generator_model,
            discriminator_fn=_shadowdata_discriminator_model,
            feat_discriminator_fn=partial(_shadowdata_feature_discriminator_model, embedded_feature_size=2,
                                          patch_count=6, is_training=True),
            image_x=images_x,
            image_y=images_y)

    def define_loss(self, model):
        # Define CycleGAN loss.
        return dcl_gan_loss(model, generator_loss_fn=tuple_losses.least_squares_generator_loss,
                            discriminator_loss_fn=tuple_losses.least_squares_discriminator_loss,
                            gen_discriminator_loss_fn=partial(contrastive_gen_data_x_loss, tau=self._tau,
                                                              batch_size=self._batch_size),
                            identity_discriminator_loss_fn=partial(contrastive_identity_loss, tau=self._tau,
                                                                   batch_size=self._batch_size),
                            nce_loss_weight=self._nce_loss_weight,
                            nce_identity_loss_weight=self._identity_loss_weight)

    def define_train_ops(self, model, loss, max_number_of_steps, **kwargs):
        gen_dis_lr = _get_lr(kwargs["gen_discriminator_lr"], max_number_of_steps)
        gen_lr = _get_lr(kwargs["generator_lr"], max_number_of_steps)
        dis_lr = _get_lr(kwargs["discriminator_lr"], max_number_of_steps)

        gen_opt = AdamOptimizer(gen_lr, beta1=0.5, use_locking=True)
        dis_opt = AdamOptimizer(dis_lr, beta1=0.5, use_locking=True)
        gen_dis_opt = AdamOptimizer(gen_dis_lr, beta1=0.5, use_locking=True)

        train_ops_x2y = cut_train_ops(
            model.model_x2y,
            loss.loss_x2y,
            generator_optimizer=gen_opt,
            discriminator_optimizer=dis_opt,
            gen_discriminator_optimizer=gen_dis_opt,
            summarize_gradients=True,
            colocate_gradients_with_ops=True,
            check_for_unused_update_ops=False,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        train_ops_y2x = cut_train_ops(
            model.model_y2x,
            loss.loss_y2x,
            generator_optimizer=gen_opt,
            discriminator_optimizer=dis_opt,
            gen_discriminator_optimizer=gen_dis_opt,
            summarize_gradients=True,
            colocate_gradients_with_ops=True,
            check_for_unused_update_ops=False,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        tf.compat.v1.summary.scalar("generator_lr", gen_lr)
        tf.compat.v1.summary.scalar("discriminator_lr", dis_lr)
        tf.compat.v1.summary.scalar("generator_discriminator_lr", gen_dis_lr)
        return DCLGANTrainOps(x2y_ops=train_ops_x2y, y2x_ops=train_ops_y2x,
                              global_step_inc_op=train_ops_x2y.global_step_inc_op,
                              train_hooks=train_ops_x2y.train_hooks)

    def get_train_hooks_fn(self):
        return get_sequential_train_hooks_dclgan(CUTTrainSteps(1, 1, 1))

    def create_validation_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                               validation_iteration_count, validation_sample_count):
        model_for_validation, x_input_tensor, y_input_tensor = define_val_model(self, data_set)
        return create_base_validation_hook(data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                                           validation_iteration_count, validation_sample_count,
                                           model_for_validation.model_x2y.generated_data,
                                           model_for_validation.model_y2x.generated_data,
                                           x_input_tensor, y_input_tensor)


class DCLGANInferenceWrapper(CycleGANInferenceWrapper):
    def __init__(self) -> None:
        super().__init__()
