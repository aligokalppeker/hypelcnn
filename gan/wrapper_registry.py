from gan.wrappers.cut_wrapper import CUTInferenceWrapper, CUTWrapper
from gan.wrappers.cycle_gan_wrapper import CycleGANInferenceWrapper, CycleGANWrapper
from gan.wrappers.dcl_gan_wrapper import DCLGANInferenceWrapper, DCLGANWrapper
from gan.wrappers.gan_wrapper import GANInferenceWrapper, GANWrapper


def get_infer_wrapper_dict():
    gan_inference_wrapper_dict = {"cycle_gan": CycleGANInferenceWrapper(),
                                  "gan_x2y": GANInferenceWrapper(fetch_shadows=False),
                                  "gan_y2x": GANInferenceWrapper(fetch_shadows=True),
                                  "cut_x2y": CUTInferenceWrapper(fetch_shadows=False),
                                  "cut_y2x": CUTInferenceWrapper(fetch_shadows=True),
                                  "dcl_gan": DCLGANInferenceWrapper()}
    return gan_inference_wrapper_dict


def get_wrapper_dict(flags):
    gan_train_wrapper_dict = {
        "cycle_gan": CycleGANWrapper(cycle_consistency_loss_weight=flags.cycle_consistency_loss_weight,
                                     identity_loss_weight=flags.identity_loss_weight,
                                     use_identity_loss=flags.use_identity_loss),
        "gan_x2y": GANWrapper(identity_loss_weight=flags.identity_loss_weight,
                              use_identity_loss=flags.use_identity_loss,
                              swap_inputs=False),
        "gan_y2x": GANWrapper(identity_loss_weight=flags.identity_loss_weight,
                              use_identity_loss=flags.use_identity_loss,
                              swap_inputs=True),
        "cut_x2y": CUTWrapper(nce_loss_weight=flags.nce_loss_weight,
                              identity_loss_weight=flags.identity_loss_weight,
                              use_identity_loss=flags.use_identity_loss,
                              swap_inputs=False,
                              tau=flags.tau,
                              embedded_feat_size=flags.embedded_feat_size,
                              patches=flags.patches,
                              batch_size=flags.batch_size),
        "cut_y2x": CUTWrapper(nce_loss_weight=flags.nce_loss_weight,
                              identity_loss_weight=flags.identity_loss_weight,
                              use_identity_loss=flags.use_identity_loss,
                              swap_inputs=True,
                              tau=flags.tau,
                              embedded_feat_size=flags.embedded_feat_size,
                              patches=flags.patches,
                              batch_size=flags.batch_size),
        "dcl_gan": DCLGANWrapper(nce_loss_weight=flags.nce_loss_weight,
                                 identity_loss_weight=flags.identity_loss_weight,
                                 use_identity_loss=flags.use_identity_loss,
                                 tau=flags.tau,
                                 embedded_feat_size=flags.embedded_feat_size,
                                 patches=flags.patches,
                                 batch_size=flags.batch_size)
    }
    return gan_train_wrapper_dict

