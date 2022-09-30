from functools import partial

from gan.shadow_data_models import shadowdata_generator_model, shadowdata_discriminator_model, \
    shadowdata_feature_discriminator_model
from gan.wrappers.cut_wrapper import CUTInferenceWrapper, CUTWrapper
from gan.wrappers.cycle_gan_wrapper import CycleGANInferenceWrapper, CycleGANWrapper
from gan.wrappers.dcl_gan_wrapper import DCLGANInferenceWrapper, DCLGANWrapper
from gan.wrappers.gan_wrapper import GANInferenceWrapper, GANWrapper


def get_infer_wrapper_dict():
    generator_fn = partial(shadowdata_generator_model, create_only_encoder=False, is_training=False)
    gan_inference_wrapper_dict = {
        "cycle_gan": CycleGANInferenceWrapper(shadow_generator_fn=generator_fn),
        "gan_x2y": GANInferenceWrapper(fetch_shadows=False, shadow_generator_fn=generator_fn),
        "gan_y2x": GANInferenceWrapper(fetch_shadows=True, shadow_generator_fn=generator_fn),
        "cut_x2y": CUTInferenceWrapper(fetch_shadows=False, shadow_generator_fn=generator_fn),
        "cut_y2x": CUTInferenceWrapper(fetch_shadows=True, shadow_generator_fn=generator_fn),
        "dcl_gan": DCLGANInferenceWrapper(shadow_generator_fn=generator_fn)}
    return gan_inference_wrapper_dict


def get_wrapper_dict(flags):
    generator_fn = partial(shadowdata_generator_model, is_training=True)
    generator_fn_gan = partial(shadowdata_generator_model, create_only_encoder=False, is_training=True)
    discriminator_fn = partial(shadowdata_discriminator_model, is_training=True, scale=flags.discriminator_reg_scale)
    feat_discriminator_fn = partial(shadowdata_feature_discriminator_model,
                                    embedded_feature_size=flags.embedded_feat_size,
                                    patch_count=flags.patches, is_training=True)

    gan_train_wrapper_dict = {
        "cycle_gan": CycleGANWrapper(cycle_consistency_loss_weight=flags.cycle_consistency_loss_weight,
                                     identity_loss_weight=flags.identity_loss_weight,
                                     use_identity_loss=flags.use_identity_loss,
                                     generator_fn=generator_fn_gan,
                                     discriminator_fn=discriminator_fn),
        "gan_x2y": GANWrapper(identity_loss_weight=flags.identity_loss_weight,
                              use_identity_loss=flags.use_identity_loss,
                              swap_inputs=False,
                              generator_fn=generator_fn_gan,
                              discriminator_fn=discriminator_fn),
        "gan_y2x": GANWrapper(identity_loss_weight=flags.identity_loss_weight,
                              use_identity_loss=flags.use_identity_loss,
                              swap_inputs=True,
                              generator_fn=generator_fn_gan,
                              discriminator_fn=discriminator_fn),
        "cut_x2y": CUTWrapper(nce_loss_weight=flags.nce_loss_weight,
                              identity_loss_weight=flags.identity_loss_weight,
                              use_identity_loss=flags.use_identity_loss,
                              swap_inputs=False,
                              tau=flags.tau,
                              batch_size=flags.batch_size,
                              generator_fn=generator_fn,
                              discriminator_fn=discriminator_fn,
                              feat_discriminator_fn=feat_discriminator_fn),
        "cut_y2x": CUTWrapper(nce_loss_weight=flags.nce_loss_weight,
                              identity_loss_weight=flags.identity_loss_weight,
                              use_identity_loss=flags.use_identity_loss,
                              swap_inputs=True,
                              tau=flags.tau,
                              batch_size=flags.batch_size,
                              generator_fn=generator_fn,
                              discriminator_fn=discriminator_fn,
                              feat_discriminator_fn=feat_discriminator_fn),
        "dcl_gan": DCLGANWrapper(nce_loss_weight=flags.nce_loss_weight,
                                 identity_loss_weight=flags.identity_loss_weight,
                                 use_identity_loss=flags.use_identity_loss,
                                 tau=flags.tau,
                                 batch_size=flags.batch_size,
                                 generator_fn=generator_fn,
                                 discriminator_fn=discriminator_fn,
                                 feat_discriminator_fn=feat_discriminator_fn)
    }
    return gan_train_wrapper_dict
