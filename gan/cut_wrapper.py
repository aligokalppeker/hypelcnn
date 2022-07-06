from __future__ import division, absolute_import, print_function

import collections
import inspect

import tensorflow as tf
from tensorflow import reduce_mean
from tensorflow_core.contrib import slim
from tensorflow_core.python.training.adam import AdamOptimizer
from tensorflow_gan.python.contrib_utils import get_trainable_variables, create_train_op
from tensorflow_gan.python.losses import tuple_losses
from tensorflow_gan.python.losses.tuple_losses import args_to_gan_model
from tensorflow_gan.python.train import _convert_tensor_or_l_or_d, RunTrainOpsHook, gan_loss

from gan_common import ValidationHook, input_x_tensor_name, input_y_tensor_name, model_base_name, model_generator_name, \
    adj_shadow_ratio, _get_lr
from shadow_data_generator import _shadowdata_generator_model, _shadowdata_discriminator_model, \
    _shadowdata_feature_discriminator_model


class CUTTrainSteps(
    collections.namedtuple('CUTTrainSteps', (
            'generator_train_steps',
            'discriminator_train_steps',
            'gen_discriminator_train_steps'
    ))):
    """Contains configuration for the CUT Training.

    Args:
      generator_train_steps: Number of generator steps to take in each GAN step.
      discriminator_train_steps: Number of discriminator steps to take in each GAN step.
    """


class CUTLoss(collections.namedtuple("CUTLoss", (
        "generator_loss",
        "discriminator_loss",
        "gen_discriminator_loss",
        "identity_discriminator_loss"))):
    """CUTLoss contains the generator, discriminator and generator discriminator losses.

    Args:
      generator_loss: A tensor for the generator loss.
      discriminator_loss: A tensor for the discriminator loss.
      gen_discriminator_loss: A tensor for the generator discriminator loss.
      identity_discriminator_loss: A tensor for the generator identity loss.
    """


class CUTTrainOps(
    collections.namedtuple("CUTTrainOps", (
            "generator_train_op",
            "discriminator_train_op",
            "gen_discriminator_train_op",
            "global_step_inc_op",
            "train_hooks"
    ))):
    """CUTTrainOps contains the training ops.

    Args:
      generator_train_op: Op that performs a generator update step.
      discriminator_train_op: Op that performs a discriminator update step.
      global_step_inc_op: Op that increments the shared global step.
      train_hooks: a list or tuple containing hooks related to training that need
        to be populated when training ops are instantiated. Used primarily for
        sync hooks.
    """

    def __new__(cls, generator_train_op, discriminator_train_op, gen_discriminator_train_op,
                global_step_inc_op, train_hooks=()):
        return super(CUTTrainOps, cls).__new__(cls, generator_train_op,
                                               discriminator_train_op, gen_discriminator_train_op,
                                               global_step_inc_op, train_hooks)


def get_sequential_train_hooks_cut(train_steps):
    """Returns a hooks function for sequential GAN training.

    Args:
      train_steps: A `GANTrainSteps` tuple that determines how many generator
        and discriminator training steps to take.

    Returns:
      A function that takes a GANTrainOps tuple and returns a list of hooks.
    """

    def get_hooks(train_ops):
        generator_hook = RunTrainOpsHook(train_ops.generator_train_op,
                                         train_steps.generator_train_steps)
        discriminator_hook = RunTrainOpsHook(train_ops.discriminator_train_op,
                                             train_steps.discriminator_train_steps)
        gen_discriminator_hook = RunTrainOpsHook(train_ops.gen_discriminator_train_op,
                                                 train_steps.gen_discriminator_train_steps)
        return [generator_hook, discriminator_hook, gen_discriminator_hook] + list(train_ops.train_hooks)

    return get_hooks


def cut_loss(
        # GANModel.
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
      add_summaries: Whether or not to add summaries for the losses.

    Returns:
      A GANLoss 2-tuple of (generator_loss, discriminator_loss). Includes
      regularization losses.

    Raises:
      ValueError: If any of the auxiliary loss weights is provided and negative.
      ValueError: If `mutual_information_penalty_weight` is provided, but the
        `model` isn't an `InfoGANModel`.
    """
    gan_loss_result = gan_loss(model=model,
                               generator_loss_fn=generator_loss_fn,
                               discriminator_loss_fn=discriminator_loss_fn,
                               gradient_penalty_weight=gradient_penalty_weight,
                               gradient_penalty_epsilon=gradient_penalty_epsilon,
                               gradient_penalty_target=gradient_penalty_target,
                               gradient_penalty_one_sided=gradient_penalty_one_sided,
                               mutual_information_penalty_weight=mutual_information_penalty_weight,
                               aux_cond_generator_weight=aux_cond_generator_weight,
                               aux_cond_discriminator_weight=aux_cond_discriminator_weight,
                               tensor_pool_fn=None,
                               reduction=reduction,
                               add_summaries=add_summaries)

    # Create standard losses with optional kwargs, if the loss functions accept
    # them.
    def _optional_kwargs(fn, possible_kwargs):
        """Returns a kwargs dictionary of valid kwargs for a given function."""
        if inspect.getargspec(fn).keywords is not None:
            return possible_kwargs
        actual_args = inspect.getargspec(fn).args
        actual_kwargs = {}
        for k, v in possible_kwargs.items():
            if k in actual_args:
                actual_kwargs[k] = v
        return actual_kwargs

    possible_kwargs = {'reduction': reduction, 'add_summaries': add_summaries}
    gen_dis_loss = gen_discriminator_loss_fn(
        model, **_optional_kwargs(generator_loss_fn, possible_kwargs))

    identity_dis_loss = identity_discriminator_loss_fn(
        model, **_optional_kwargs(generator_loss_fn, possible_kwargs))

    if model.feat_discriminator_gen_data_scope:
        gen_dis_reg_loss = tf.compat.v1.losses.get_regularization_loss(
            model.feat_discriminator_gen_data_scope.name)
    else:
        gen_dis_reg_loss = 0

    return CUTLoss(gan_loss_result.generator_loss +
                   (nce_loss_weight * gen_dis_loss) + (nce_identity_loss_weight * identity_dis_loss),
                   gan_loss_result.discriminator_loss,
                   gen_dis_loss + gen_dis_reg_loss,
                   identity_dis_loss + gen_dis_reg_loss)


class CUTModel(
    collections.namedtuple("CUTModel", (
            "generator_inputs",
            "generated_data",
            "generator_variables",
            "generator_scope",
            "generator_fn",
            "real_data",
            "discriminator_real_outputs",
            "discriminator_gen_outputs",
            "discriminator_variables",
            "discriminator_scope",
            "discriminator_fn",
            "feat_discriminator_gen_data",
            "feat_discriminator_gen_data_variables",
            "feat_discriminator_gen_data_scope",
            "feat_discriminator_gen_data_fn",
            "feat_discriminator_real_data_x",
            "feat_discriminator_real_data_y",
            "feat_discriminator_generated_data_y",
    ))):
    """A CUTModel contains all the pieces needed for GAN training.

    Generative Adversarial Networks (https://arxiv.org/abs/1406.2661) attempt
    to create an implicit generative model of data by solving a two agent game.
    The generator generates candidate examples that are supposed to match the
    data distribution, and the discriminator aims to tell the real examples
    apart from the generated samples.

    Args:
      generator_inputs: The random noise source that acts as input to the
        generator.
      generated_data: The generated output data of the GAN.
      generator_variables: A list of all generator variables.
      generator_scope: Variable scope all generator variables live in.
      generator_fn: The generator function.
      real_data: A tensor or real data.
      discriminator_real_outputs: The discriminator's output on real data.
      discriminator_gen_outputs: The discriminator's output on generated data.
      discriminator_variables: A list of all discriminator variables.
      discriminator_scope: Variable scope all discriminator variables live in.
      discriminator_fn: The discriminator function.
    """


def cut_model(
        # Lambdas defining models.
        generator_fn,
        discriminator_fn,
        feat_discriminator_fn,
        # Real data and conditioning.
        real_data,
        generator_inputs,
        # Optional scopes.
        generator_scope="Generator",
        discriminator_scope="Discriminator",
        feat_discriminator_scope="FeatDiscriminator",
        # Options.
        check_shapes=True):
    """Returns GAN model outputs and variables.

    Args:
      feat_discriminator_scope:
      generator_fn: A python lambda that takes `generator_inputs` as inputs and
        returns the outputs of the GAN generator.
      discriminator_fn: A python lambda that takes `real_data`/`generated data`
        and `generator_inputs`. Outputs a Tensor in the range [-inf, inf].
      real_data: A Tensor representing the real data.
      generator_inputs: A Tensor or list of Tensors to the generator. In the
        vanilla GAN case, this might be a single noise Tensor. In the conditional
        GAN case, this might be the generator's conditioning.
      generator_scope: Optional generator variable scope. Useful if you want to
        reuse a subgraph that has already been created.
      discriminator_scope: Optional discriminator variable scope. Useful if you
        want to reuse a subgraph that has already been created.
      check_shapes: If `True`, check that generator produces Tensors that are the
        same shape as real data. Otherwise, skip this check.

    Returns:
      A GANModel namedtuple.

    Raises:
      ValueError: If the generator outputs a Tensor that isn't the same shape as
        `real_data`.
      ValueError: If TF is executing eagerly.
    """
    if tf.executing_eagerly():
        raise ValueError('`vut_model` doesn\'t work when executing eagerly.')
    # Create models
    with tf.compat.v1.variable_scope(generator_scope, reuse=tf.compat.v1.AUTO_REUSE) as gen_scope:
        generator_inputs = _convert_tensor_or_l_or_d(generator_inputs)
        generated_data = generator_fn(generator_inputs)
    with tf.compat.v1.variable_scope(discriminator_scope, reuse=tf.compat.v1.AUTO_REUSE) as dis_scope:
        discriminator_gen_outputs = discriminator_fn(generated_data, generator_inputs)
    with tf.compat.v1.variable_scope(dis_scope, reuse=True):
        real_data = _convert_tensor_or_l_or_d(real_data)
        discriminator_real_outputs = discriminator_fn(real_data, generator_inputs)
    # Create cut models
    # generated data from x
    with tf.compat.v1.variable_scope(gen_scope, reuse=True):
        feature_embeddings_generated_data = generator_fn(generated_data, create_only_encoder=True)
    with tf.compat.v1.variable_scope(feat_discriminator_scope,
                                     reuse=tf.compat.v1.AUTO_REUSE) as generated_data_feat_dis_scope:
        feat_discriminator_generated_data = feat_discriminator_fn(feature_embeddings_generated_data)

    # real data from x
    with tf.compat.v1.variable_scope(gen_scope, reuse=True):
        feature_embeddings_real_data_x = generator_fn(generator_inputs, create_only_encoder=True)
    with tf.compat.v1.variable_scope(feat_discriminator_scope, reuse=tf.compat.v1.AUTO_REUSE):
        feat_discriminator_real_data_x = feat_discriminator_fn(feature_embeddings_real_data_x)

    # real data from y
    with tf.compat.v1.variable_scope(gen_scope, reuse=True):
        feature_embeddings_real_data_y = generator_fn(real_data, create_only_encoder=True)
    with tf.compat.v1.variable_scope(feat_discriminator_scope, reuse=tf.compat.v1.AUTO_REUSE):
        feat_discriminator_real_data_y = feat_discriminator_fn(feature_embeddings_real_data_y)

    # real_data fed into generator(identity data)
    with tf.compat.v1.variable_scope(gen_scope, reuse=True):
        feature_embeddings_generated_data_y = generator_fn(real_data, create_only_encoder=True)
    with tf.compat.v1.variable_scope(feat_discriminator_scope, reuse=tf.compat.v1.AUTO_REUSE):
        feat_discriminator_generated_data_y = feat_discriminator_fn(feature_embeddings_generated_data_y)

    if check_shapes:
        if not generated_data.shape.is_compatible_with(real_data.shape):
            raise ValueError(
                'Generator output shape (%s) must be the same shape as real data '
                '(%s).' % (generated_data.shape, real_data.shape))

    # Get model-specific variables.
    generator_variables = get_trainable_variables(gen_scope)
    discriminator_variables = get_trainable_variables(dis_scope)
    feat_disc_gen_inputs_variables = get_trainable_variables(generated_data_feat_dis_scope)

    return CUTModel(
        generator_inputs, generated_data, generator_variables, gen_scope, generator_fn,
        real_data,
        discriminator_real_outputs, discriminator_gen_outputs, discriminator_variables, dis_scope, discriminator_fn,
        feat_discriminator_generated_data, feat_disc_gen_inputs_variables, generated_data_feat_dis_scope,
        feat_discriminator_fn,
        feat_discriminator_real_data_x,
        feat_discriminator_real_data_y,
        feat_discriminator_generated_data_y)


# Contrastive loss for x data and generated x data.
def contrastive_gen_data_x_loss_impl(
        feat_discriminator_gen_data,
        feat_discriminator_real_data_x,
        discriminator_gen_outputs,
        weights=1.0,
        scope=None,
        loss_collection=tf.compat.v1.GraphKeys.LOSSES,
        reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    # Override parameter, it should be always SUM_OVER_BATCH_SIZE
    reduction = tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE
    with tf.compat.v1.name_scope(scope, "constrastive_gen_loss", (discriminator_gen_outputs, weights)) as scope:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(tf.nn.softmax(feat_discriminator_real_data_x),
                                                          feat_discriminator_gen_data)
        loss = tf.compat.v1.losses.compute_weighted_loss(loss, weights, scope,
                                                         loss_collection, reduction)
        # softmax_entropy = tf.boolean_mask(softmax_entropy, tf.is_finite(softmax_entropy))
        # patch_nce_loss = tf.reduce_mean(softmax_entropy)
        # patch_nce_loss = tf.where(tf.logical_not(tf.is_finite(patch_nce_loss)), tf.zeros_like(patch_nce_loss),
        #                           patch_nce_loss)
    if add_summaries:
        tf.compat.v1.summary.scalar("nce_loss_data_x", loss)
    return loss


contrastive_gen_data_x_loss = args_to_gan_model(contrastive_gen_data_x_loss_impl)


# Contrastive loss for x data and generated x data.
def contrastive_identity_loss_impl(
        feat_discriminator_real_data_y,
        feat_discriminator_generated_data_y,
        discriminator_gen_outputs,
        weights=1.0,
        scope=None,
        loss_collection=tf.compat.v1.GraphKeys.LOSSES,
        reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    # Override parameter, it should be always SUM_OVER_BATCH_SIZE
    reduction = tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE
    with tf.compat.v1.name_scope(scope, "constrastive_identity_loss", (discriminator_gen_outputs, weights)) as scope:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(tf.nn.softmax(feat_discriminator_real_data_y),
                                                          feat_discriminator_generated_data_y)
        loss = tf.compat.v1.losses.compute_weighted_loss(loss, weights, scope,
                                                         loss_collection, reduction)
    if add_summaries:
        tf.compat.v1.summary.scalar("nce_loss_identity", loss)
    return loss


contrastive_identity_loss = args_to_gan_model(contrastive_identity_loss_impl)


class CUTWrapper:

    def __init__(self, nce_loss_weight, identity_loss_weight, use_identity_loss, swap_inputs) -> None:
        super().__init__()
        self._nce_loss_weight = nce_loss_weight
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

        gan_model = cut_model(
            generator_fn=_shadowdata_generator_model,
            discriminator_fn=_shadowdata_discriminator_model,
            feat_discriminator_fn=_shadowdata_feature_discriminator_model,
            generator_inputs=generator_inputs,
            real_data=real_data)

        # Add summaries for generated images.
        # tfgan.eval.add_cyclegan_image_summaries(gan_model)
        return gan_model

    def define_loss(self, model):
        # Define CycleGAN loss.
        loss = cut_loss(model, generator_loss_fn=tuple_losses.least_squares_generator_loss,
                        discriminator_loss_fn=tuple_losses.least_squares_discriminator_loss,
                        gen_discriminator_loss_fn=contrastive_gen_data_x_loss,
                        identity_discriminator_loss_fn=contrastive_identity_loss,
                        nce_loss_weight=self._nce_loss_weight,
                        nce_identity_loss_weight=self._identity_loss_weight)
        return loss

    @staticmethod
    def _get_update_ops(kwargs, gen_scope, dis_scope, gen_dis_scope, check_for_unused_ops=True):
        """Gets generator and discriminator update ops.

        Args:
          kwargs: A dictionary of kwargs to be passed to `create_train_op`.
            `update_ops` is removed, if present.
          gen_scope: A scope for the generator.
          dis_scope: A scope for the discriminator.
          check_for_unused_ops: A Python bool. If `True`, throw Exception if there are
            unused update ops.

        Returns:
          A 3-tuple of (generator update ops, discriminator train ops, generator discriminator train ops).

        Raises:
          ValueError: If there are update ops outside of the generator or
            discriminator scopes.
        """
        if 'update_ops' in kwargs:
            update_ops = set(kwargs['update_ops'])
            del kwargs['update_ops']
        else:
            update_ops = set(
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))

        all_gen_ops = set(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, gen_scope))
        all_dis_ops = set(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, dis_scope))
        all_gen_dis_ops = set(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, gen_dis_scope))

        if check_for_unused_ops:
            unused_ops = update_ops - all_gen_ops - all_dis_ops - all_gen_dis_ops
            if unused_ops:
                raise ValueError('There are unused update ops: %s' % unused_ops)

        gen_update_ops = list(all_gen_ops & update_ops)
        dis_update_ops = list(all_dis_ops & update_ops)
        gen_dis_update_ops = list(all_gen_dis_ops & update_ops)

        return gen_update_ops, dis_update_ops, gen_dis_update_ops

    @staticmethod
    def _cut_train_ops(
            model,
            loss,
            generator_optimizer,
            discriminator_optimizer,
            gen_discriminator_optimizer,
            check_for_unused_update_ops=True,
            is_chief=True,
            # Optional args to pass directly to the `create_train_op`.
            **kwargs):
        """Returns GAN train ops.

        The highest-level call in TF-GAN. It is composed of functions that can also
        be called, should a user require more control over some part of the GAN
        training process.

        Args:
          model: A GANModel.
          loss: A GANLoss.
          generator_optimizer: The optimizer for generator updates.
          discriminator_optimizer: The optimizer for the discriminator updates.
          gen_discriminator_optimizer: The optimizer for the generator discriminator updates.
          check_for_unused_update_ops: If `True`, throws an exception if there are
            update ops outside of the generator or discriminator scopes.
          is_chief: Specifies whether or not the training is being run by the primary
            replica during replica training.
          **kwargs: Keyword args to pass directly to
            `training.create_train_op` for both the generator and
            discriminator train op.

        Returns:
          A GANTrainOps tuple of (generator_train_op, discriminator_train_op) that can
          be used to train a generator/discriminator pair.
        """
        # Create global step increment op.
        global_step = tf.compat.v1.train.get_or_create_global_step()
        global_step_inc = global_step.assign_add(1)

        # Get generator and discriminator update ops. We split them so that update
        # ops aren't accidentally run multiple times. For now, throw an error if
        # there are update ops that aren't associated with either the generator or
        # the discriminator. Might modify the `kwargs` dictionary.
        gen_update_ops, dis_update_ops, gen_dis_update_ops = CUTWrapper._get_update_ops(
            kwargs,
            model.generator_scope.name, model.discriminator_scope.name, model.feat_discriminator_gen_data_scope.name,
            check_for_unused_update_ops)

        # Get the sync hooks if these are needed.
        sync_hooks = []

        generator_global_step = None
        if isinstance(generator_optimizer, tf.compat.v1.train.SyncReplicasOptimizer):
            # WARNING: Making this variable a local variable causes sync replicas to
            # hang forever.
            generator_global_step = tf.compat.v1.get_variable(
                'dummy_global_step_generator',
                shape=[],
                dtype=global_step.dtype.base_dtype,
                initializer=tf.compat.v1.initializers.zeros(),
                trainable=False,
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES])
            gen_update_ops += [generator_global_step.assign(global_step)]
            sync_hooks.append(generator_optimizer.make_session_run_hook(is_chief))
        with tf.compat.v1.name_scope('generator_train'):
            gen_train_op = create_train_op(
                total_loss=loss.generator_loss,
                optimizer=generator_optimizer,
                variables_to_train=model.generator_variables,
                global_step=generator_global_step,
                update_ops=gen_update_ops,
                check_numerics=False,
                **kwargs)

        discriminator_global_step = None
        if isinstance(discriminator_optimizer, tf.compat.v1.train.SyncReplicasOptimizer):
            # See comment above `generator_global_step`.
            discriminator_global_step = tf.compat.v1.get_variable(
                'dummy_global_step_discriminator',
                shape=[],
                dtype=global_step.dtype.base_dtype,
                initializer=tf.compat.v1.initializers.zeros(),
                trainable=False,
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES])
            dis_update_ops += [discriminator_global_step.assign(global_step)]
            sync_hooks.append(discriminator_optimizer.make_session_run_hook(is_chief))
        with tf.compat.v1.name_scope('discriminator_train'):
            disc_train_op = create_train_op(
                total_loss=loss.discriminator_loss,
                optimizer=discriminator_optimizer,
                variables_to_train=model.discriminator_variables,
                global_step=discriminator_global_step,
                update_ops=dis_update_ops,
                check_numerics=False,
                **kwargs)

        gen_discriminator_global_step = None
        if isinstance(gen_discriminator_optimizer, tf.compat.v1.train.SyncReplicasOptimizer):
            # See comment above `generator_global_step`.
            gen_discriminator_global_step = tf.compat.v1.get_variable(
                'dummy_global_step_discriminator',
                shape=[],
                dtype=global_step.dtype.base_dtype,
                initializer=tf.compat.v1.initializers.zeros(),
                trainable=False,
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES])
            gen_dis_update_ops += [gen_discriminator_global_step.assign(global_step)]
            sync_hooks.append(gen_discriminator_optimizer.make_session_run_hook(is_chief))
        with tf.compat.v1.name_scope('gen_discriminator_train'):
            gen_disc_train_op = create_train_op(
                total_loss=loss.gen_discriminator_loss,
                optimizer=gen_discriminator_optimizer,
                variables_to_train=model.feat_discriminator_gen_data_variables,
                global_step=gen_discriminator_global_step,
                update_ops=gen_dis_update_ops,
                check_numerics=False,
                **kwargs)

        return CUTTrainOps(gen_train_op, disc_train_op, gen_disc_train_op, global_step_inc, sync_hooks)

    def define_train_ops(self, model, loss, max_number_of_steps, **kwargs):
        gen_dis_lr = _get_lr(kwargs["gen_discriminator_lr"], max_number_of_steps)
        gen_lr = _get_lr(kwargs["generator_lr"], max_number_of_steps)
        dis_lr = _get_lr(kwargs["discriminator_lr"], max_number_of_steps)

        gen_opt = AdamOptimizer(gen_lr, beta1=0.5, use_locking=True)
        dis_opt = AdamOptimizer(dis_lr, beta1=0.5, use_locking=True)
        gen_dis_opt = AdamOptimizer(gen_dis_lr, beta1=0.5, use_locking=True)

        train_ops = self._cut_train_ops(
            model,
            loss,
            generator_optimizer=gen_opt,
            discriminator_optimizer=dis_opt,
            gen_discriminator_optimizer=gen_dis_opt,
            summarize_gradients=True,
            colocate_gradients_with_ops=True,
            check_for_unused_update_ops=False,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        tf.summary.scalar("generator_lr", gen_lr)
        tf.summary.scalar("discriminator_lr", dis_lr)
        tf.summary.scalar("generator_discriminator_lr", gen_dis_lr)

        return train_ops

    def get_train_hooks_fn(self):
        return get_sequential_train_hooks_cut(CUTTrainSteps(1, 1, 1))

    def create_validation_hook(self, data_set, loader, log_dir, neighborhood, shadow_map, shadow_ratio,
                               validation_iteration_count, validation_sample_count):
        element_size = data_set.get_data_shape()
        element_size = [None, element_size[0], element_size[1], data_set.get_casi_band_count()]

        x_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name=input_x_tensor_name)
        y_input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size, name=input_y_tensor_name)
        model_for_validation = self.define_model(x_input_tensor, y_input_tensor)
        shadowed_validation_hook = ValidationHook(iteration_freq=validation_iteration_count,
                                                  sample_count=validation_sample_count,
                                                  log_dir=log_dir,
                                                  loader=loader, data_set=data_set, neighborhood=neighborhood,
                                                  shadow_map=shadow_map,
                                                  shadow_ratio=adj_shadow_ratio(shadow_ratio, self._swap_inputs),
                                                  input_tensor=y_input_tensor if self._swap_inputs else x_input_tensor,
                                                  infer_model=model_for_validation.generated_data,
                                                  fetch_shadows=False,
                                                  name_suffix="deshadowed" if self._swap_inputs else "shadowed")
        return shadowed_validation_hook


class CUTInferenceWrapper:
    def __init__(self, fetch_shadows):
        self.fetch_shadows = fetch_shadows

    def construct_inference_graph(self, input_tensor, is_shadow_graph, clip_invalid_values=False):
        shp = input_tensor.get_shape()

        output_tensor_in_col = []
        for first_dim in range(shp[1]):
            output_tensor_in_row = []
            for second_dim in range(shp[2]):
                input_cell = tf.expand_dims(tf.expand_dims(input_tensor[:, first_dim, second_dim], 0), 0)
                with tf.variable_scope(model_base_name):
                    with tf.variable_scope(model_generator_name, reuse=tf.AUTO_REUSE):
                        generated_tensor = _shadowdata_generator_model(input_cell, False)
                        if clip_invalid_values:
                            input_mean = reduce_mean(input_cell)
                            generated_mean = reduce_mean(generated_tensor)

                if clip_invalid_values:
                    result_tensor = tf.cond(tf.less(generated_mean, input_mean),
                                            lambda: generated_tensor,
                                            lambda: input_cell)
                else:
                    result_tensor = generated_tensor

                output_tensor_in_row.append(tf.squeeze(result_tensor, [0, 1]))
            image_output_row = tf.concat(output_tensor_in_row, axis=0)
            output_tensor_in_col.append(image_output_row)

        image_output_row = tf.stack(output_tensor_in_col)

        return image_output_row

    @staticmethod
    def __create_input_tensor(data_set, is_shadow_graph):
        element_size = data_set.get_data_shape()
        element_size = [None, element_size[0], element_size[1], data_set.get_casi_band_count()]
        input_tensor = tf.placeholder(dtype=tf.float32, shape=element_size,
                                      name=input_x_tensor_name if is_shadow_graph else input_y_tensor_name)
        return input_tensor

    def make_inference_graph(self, data_set, is_shadow_graph, clip_invalid_values):
        input_tensor = self.__create_input_tensor(data_set, is_shadow_graph)
        generated = self.construct_inference_graph(input_tensor, is_shadow_graph, clip_invalid_values)
        return input_tensor, generated

    def create_generator_restorer(self):
        # Restore all the variables that were saved in the checkpoint.
        gan_restorer = tf.train.Saver(
            slim.get_variables_to_restore(include=[model_base_name]),
            name='GeneratorRestoreHandler'
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
                              infer_model=self.construct_inference_graph(input_tensor, None),
                              fetch_shadows=self.fetch_shadows, name_suffix="")
