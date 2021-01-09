import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy
from tifffile import imwrite

from GRSS2013DataLoader import GRSS2013DataLoader
from cmd_parser import parse_cmd
from common_nn_operations import get_class, create_target_image_via_samples, INVALID_TARGET_VALUE, create_colored_image

BUILDING_CLASS = 7
BUILDING_SHADOW_CLASS = 6


def create_shadow_corrected_image(casi_normalized, casi, shadow_map):
    ratio = GRSS2013DataLoader.calculate_shadow_ratio(casi,
                                                      shadow_map,
                                                      numpy.logical_not(shadow_map).astype(int))
    add_coef = numpy.repeat(numpy.expand_dims(shadow_map, axis=2), casi_normalized.shape[2], axis=2) * (ratio - 1)
    final_casi = casi + (casi * add_coef)
    imwrite("muulf_hsi_shadow_corrected.tif", final_casi.astype(numpy.float32), planarconfig='contig')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', nargs='?', const=True, type=str,
                        default=os.path.dirname(__file__),
                        help='Path for saving output images')
    flags = parse_cmd(parser)

    loader_name = flags.loader_name
    loader = get_class(loader_name + '.' + loader_name)(flags.path)
    sample_set = loader.load_samples(0.1)
    data_set = loader.load_data(0, True)
    scene_shape = loader.get_scene_shape(data_set)

    target_classes_as_image = create_target_image_via_samples(sample_set, scene_shape)

    shadow_map = get_shadow_map(target_classes_as_image)
    imwrite("muulf_shadow_map.tif", shadow_map, planarconfig='contig')
    create_shadow_corrected_image(data_set.casi, loader.load_data(0, False).casi, shadow_map)
    draw_targets(loader.get_target_color_list(), target_classes_as_image, "Targets")

    # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(shadow_map)
    contours, hierarchy = cv2.findContours(shadow_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # draw_im = numpy.zeros(shadow_map.shape, dtype=numpy.uint8)
    # cv2.drawContours(draw_im, contours, -1, 255, 3)
    for contour in contours:
        target_map = {}
        fill_targets_for_contour(contour, target_classes_as_image, target_map)
        if BUILDING_SHADOW_CLASS in target_map:
            del target_map[BUILDING_SHADOW_CLASS]
        if INVALID_TARGET_VALUE in target_map:
            del target_map[INVALID_TARGET_VALUE]
        if BUILDING_CLASS in target_map:
            del target_map[BUILDING_CLASS]
        final_neigh_target = None
        final_neigh_count = 0
        for neigh_target, neigh_count in target_map.items():
            if final_neigh_target is not None:
                if neigh_count > final_neigh_count:
                    final_neigh_target = neigh_target
                    final_neigh_count = neigh_count
            else:
                final_neigh_target = neigh_target
                final_neigh_count = neigh_count
        # print(final_neigh_target)
        # print(final_neigh_count)
        if final_neigh_target is None:
            print("found contour with no proper neighbors")
        else:
            image = get_contour_image(shadow_map.shape, contour)
            target_classes_as_image[image] = final_neigh_target
            print("shadow converted to neighboring target %d" % final_neigh_target)

    draw_targets(loader.get_target_color_list(), target_classes_as_image, "Targets after shadow correction")
    # increase target level as one
    target_classes_as_image[target_classes_as_image != INVALID_TARGET_VALUE] = target_classes_as_image[
                                                                                   target_classes_as_image != INVALID_TARGET_VALUE] + 1
    imwrite("muulf_gt_shadow_corrected.tif", target_classes_as_image, planarconfig='contig')


def draw_targets(color_list, target_classes_as_image, figure_name):
    plt.imshow(create_colored_image(target_classes_as_image, color_list))
    plt.title(figure_name), plt.xticks([]), plt.yticks([])
    plt.show()


def get_shadow_map(target_image):
    return (target_image == BUILDING_SHADOW_CLASS).astype(numpy.uint8)


def fill_targets_for_contour(contour, target_image, target_map):
    for index in range(0, contour.shape[0]):
        x, y = contour[index][0][1], contour[index][0][0]
        target_list = find_neighbouring_target(target_image, x, y)
        for target in target_list:
            if target in target_map:
                target_map[target] = target_map[target] + 1
            else:
                target_map[target] = 0


def center_of_contour(contour):
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy


def get_contour_image(image_shape, contour):
    draw_im = numpy.zeros(image_shape, dtype=numpy.uint8)
    draw_pix_val = 255
    return cv2.drawContours(draw_im, [contour], 0, draw_pix_val, -1) == draw_pix_val


neighborhood_pair_list = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (1, 0), (-1, 1), (-1, -1)]


def find_neighbouring_target(target_image, x, y):
    target_list = []
    for delta_x, delta_y in neighborhood_pair_list:
        target_list.append(target_image[x + delta_x, y + delta_y])
    return target_list


if __name__ == '__main__':
    main()
