# Derived from Spectral2RGB github repository
import colour
import numpy
from colour.colorimetry import MSDS_CMFS_STANDARD_OBSERVER
from colour.utilities import tsplit
from skimage.color import colorconv


def _spectral2xyz_img_vectorized(cmfs, r):
    x_bar, y_bar, z_bar = tsplit(cmfs)  #
    # illuminant.
    s = colour.SDS_ILLUMINANTS['E'].values[0:31] / 100.
    dw = 10
    k = 100 / (numpy.sum(y_bar * s) * dw)

    x_p = r * x_bar * s * dw
    y_p = r * y_bar * s * dw
    z_p = r * z_bar * s * dw

    xyz = k * numpy.sum(numpy.array([x_p, y_p, z_p]), axis=-1)
    xyz = numpy.rollaxis(xyz, 1, 0)
    return xyz


def _spectral2xyz_img(hs, cmf_name):
    h, w, c = hs.shape
    hs = hs.reshape(-1, c)
    cmfs = _get_cmfs(cmf_name=cmf_name, nm_range=(400., 701.), nm_step=10, split=False)
    xyz = _spectral2xyz_img_vectorized(cmfs, hs)  # (nb_px, 3)
    xyz = xyz.reshape((h, w, 3))
    return xyz


def _spectral2srgb_img(spectral, cmf_name):
    xyz = _spectral2xyz_img(hs=spectral, cmf_name=cmf_name)
    s_rgb = colorconv.xyz2rgb(xyz / 100.)
    return s_rgb


def _get_cmfs(cmf_name='cie1964_10', nm_range=(400., 701.), nm_step=10, split=True):
    if cmf_name == 'cie1931_2':
        cmf_full_name = 'CIE 1931 2 Degree Standard Observer'
    elif cmf_name == 'cie2012_2':
        cmf_full_name = 'CIE 2012 2 Degree Standard Observer'
    elif cmf_name == 'cie2012_10':
        cmf_full_name = 'CIE 2012 10 Degree Standard Observer'
    elif cmf_name == 'cie1964_10':
        cmf_full_name = 'CIE 1964 10 Degree Standard Observer'
    else:
        raise AttributeError('Wrong cmf name')
    cmfs = MSDS_CMFS_STANDARD_OBSERVER[cmf_full_name]

    # subsample and trim range
    ix_wl_first = numpy.where(cmfs.wavelengths == nm_range[0])[0][0]
    ix_wl_last = numpy.where(cmfs.wavelengths == nm_range[1] + 1.)[0][0]
    cmfs = cmfs.values[ix_wl_first:ix_wl_last:int(nm_step), :]

    if split:
        x_bar, y_bar, z_bar = tsplit(cmfs)
        return x_bar, y_bar, z_bar
    else:
        return cmfs


def get_rgb_from_hsi(band_measurements, casi_normalized):
    wi = numpy.round(band_measurements)
    visual_spec = list(range(400, 701, 10))
    x_cor = []
    for i in visual_spec:
        x_cor_i = numpy.where(abs(wi - i) == min(abs(wi - i)))
        x_cor.append(x_cor_i[0].tolist()[0])
    spectral_x = casi_normalized[:, :, x_cor]  # extract the wavebands in visual range
    shi_rgb = _spectral2srgb_img(spectral_x, "cie1931_2")
    return shi_rgb
