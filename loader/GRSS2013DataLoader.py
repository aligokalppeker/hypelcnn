import numpy
from tifffile import imread

from loader.DataLoader import DataLoader, SampleSet
from common_nn_operations import read_targets_from_image, shuffle_test_data_using_ratio, load_shadow_map_common, \
    DataSet, get_data_point_func
from gan.gan_utilities import create_gan_struct, create_simple_shadow_struct
from gan.wrappers.cycle_gan_wrapper import CycleGANInferenceWrapper
from gan.wrappers.gan_wrapper import GANInferenceWrapper


class GRSS2013DataLoader(DataLoader):

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, neighborhood, normalize):
        casi = imread(self.get_model_base_dir() + "2013_IEEE_GRSS_DF_Contest_CASI.tif")
        lidar = imread(self.get_model_base_dir() + "2013_IEEE_GRSS_DF_Contest_LiDAR.tif")[:, :, numpy.newaxis]
        data_set = DataSet(shadow_creator_dict=None, casi=casi, lidar=lidar, neighborhood=neighborhood,
                           normalize=normalize)

        _, shadow_ratio = self.load_shadow_map(neighborhood, data_set)
        data_set.shadow_creator_dict = {"cycle_gan": create_gan_struct(CycleGANInferenceWrapper(),
                                                                       self.get_model_base_dir(),
                                                                       "shadow_cycle_gan/modelv4/model.ckpt-33000"),
                                        "gan": create_gan_struct(GANInferenceWrapper(None),
                                                                 self.get_model_base_dir(),
                                                                 "../utilities/log/model.ckpt-203000"),
                                        "simple": create_simple_shadow_struct(shadow_ratio)}

        return data_set

    def load_shadow_map(self, neighborhood, data_set):
        return load_shadow_map_common(data_set, neighborhood, self.get_model_base_dir() + "shadow_map.tif")

    def load_samples(self, train_data_ratio, test_data_ratio):
        train_set = self.read_targets("2013_IEEE_GRSS_DF_Contest_Samples_TR.tif")
        validation_set = self.read_targets("2013_IEEE_GRSS_DF_Contest_Samples_VA.tif")

        test_set, train_set = shuffle_test_data_using_ratio(train_set, test_data_ratio)

        return SampleSet(training_targets=train_set, test_targets=test_set,
                         validation_targets=validation_set)

    def read_targets(self, target_image_path):
        targets = imread(self.get_model_base_dir() + target_image_path)
        return read_targets_from_image(targets, self.get_class_count())

    def get_class_count(self):
        return range(0, 15)

    def get_model_base_dir(self):
        return self.base_dir + '/2013_DFTC/'

    def get_point_value(self, data_set, point):
        return get_data_point_func(data_set.casi, data_set.lidar, data_set.neighborhood, point)

    def get_target_color_list(self):
        color_list = numpy.zeros([15, 3], numpy.uint8)
        # Grass Healthy
        color_list[0, :] = [0, 180, 0]
        # Grass Stressed
        color_list[1, :] = [0, 124, 0]
        # Grass Synthetic
        color_list[2, :] = [0, 137, 69]
        # Tree
        color_list[3, :] = [0, 69, 0]
        # Soil
        color_list[4, :] = [172, 125, 11]
        # Water
        color_list[5, :] = [0, 190, 194]
        # Residential
        color_list[6, :] = [120, 0, 0]
        # Commercial
        color_list[7, :] = [216, 217, 247]
        # Road
        color_list[8, :] = [121, 121, 121]
        # Highway
        color_list[9, :] = [205, 172, 127]
        # Railway
        color_list[10, :] = [220, 175, 120]
        # Parking lot 1
        color_list[11, :] = [100, 100, 100]
        # Parking lot 2
        color_list[12, :] = [185, 175, 94]
        # Tennis lot
        color_list[13, :] = [0, 237, 0]
        # Running track
        color_list[14, :] = [207, 18, 56]
        return color_list

    def get_band_measurements(self):
        return numpy.linspace(380, 1050, num=144)
