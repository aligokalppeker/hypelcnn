import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cv2 import resize, matchTemplate, minMaxLoc, rectangle, TM_CCORR_NORMED, INTER_AREA

from common.cmd_parser import add_parse_cmds_for_classification, add_parse_cmds_for_loggers
from common.common_nn_ops import get_loader_from_name


def main(_):
    parser = argparse.ArgumentParser()
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_classification(parser)
    flags, unparsed = parser.parse_known_args()

    normalize = True

    lidar_grss2013_scale = 5
    lidar_grss2018_scale = lidar_grss2013_scale / 2.5

    loader_name = "GRSS2013DataLoader"
    loader = get_loader_from_name(loader_name, flags.path)
    grss_2013_data_set = loader.load_data(0, normalize)

    loader_name = "GRSS2018DataLoader"
    loader = get_loader_from_name(loader_name, flags.path)
    grss_2018_data_set = loader.load_data(0, normalize)

    grss_2013_band = 8
    grss_2018_band = 2

    match_data(grss_2013_band, grss_2018_band,
               grss_2013_data_set, grss_2018_data_set,
               lidar_grss2013_scale, lidar_grss2018_scale)


def match_data(grss_2013_band, grss_2018_band, grss_2013_data_set, grss_2018_data_set,
               grss2013_scale, grss2018_scale):
    band_grss2013 = grss_2013_data_set.casi[:, :, grss_2013_band]
    # band_grss2013 = band_grss2013 - np.min(band_grss2013)
    band_grss2013 = resize(band_grss2013, (
        band_grss2013.shape[1] * grss2013_scale, band_grss2013.shape[0] * grss2013_scale),
                           interpolation=INTER_AREA)
    # hist plot
    hist_2013, bin_edges_2013 = np.histogram(band_grss2013, bins=np.arange(np.max(band_grss2013)))
    plt.plot(bin_edges_2013[:-1], hist_2013, label="2013")
    plt.show()

    band_grss2018 = np.squeeze(grss_2018_data_set.casi[:, :, grss_2018_band]).astype(np.float32)
    # band_grss2018 = band_grss2018 - np.min(band_grss2018)
    # band_grss2018 = band_grss2018[0:-700, 0:-150]
    band_grss2018 = band_grss2018[0:-350, 0:-75]
    band_grss2018 = resize(band_grss2018, (
        int(band_grss2018.shape[1] * grss2018_scale), int(band_grss2018.shape[0] * grss2018_scale)),
                           interpolation=INTER_AREA)
    # band_grss2018 = band_grss2018 * 1.5
    # hist plot
    hist_2018, bin_edges_2018 = np.histogram(band_grss2018, bins=np.arange(np.max(band_grss2018)))
    plt.plot(bin_edges_2018[:-1], hist_2018, label="2018")
    plt.show()

    res = matchTemplate(band_grss2013, band_grss2018, TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = minMaxLoc(res)
    w, h = band_grss2018.shape[::-1]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    im_normalized_2018 = (band_grss2018 / np.max(band_grss2018) * 255).astype('uint8')
    im_normalized_2013 = (band_grss2013 / np.max(band_grss2013) * 255).astype('uint8')
    # imwrite("image2013.png", im_normalized_2013)
    # imwrite("image2018.png", im_normalized_2018)
    rectangle(im_normalized_2013, top_left, bottom_right, 255, 4 * grss2013_scale)
    plt.imshow(im_normalized_2013)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()
    print("Top Left", top_left)
    print("Top Left(scaled) (%f, %f)" % (
        top_left[0] / grss2013_scale, top_left[1] / grss2013_scale))
    print("Bottom Right", bottom_right)
    print("Bottom Right(scaled) (%f, %f)" % (
        bottom_right[0] / grss2013_scale, bottom_right[1] / grss2013_scale))


if __name__ == '__main__':
    tf.app.run(main=main)
