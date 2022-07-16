import os
from distutils.util import strtobool


def type_ensure_strtobool(val):
    return bool(strtobool(str(val)))

def add_parse_cmds_for_classification(parser):
    add_parse_cmds_for_loader(parser)
    add_parse_cmds_for_loggers(parser)
    add_parse_cmds_for_trainers(parser)

    parser.add_argument("--importer_name", nargs="?", const=True, type=str,
                        default="InMemoryImporter",
                        help="Data set loader name, Values : GRSS2013DataLoader")
    parser.add_argument("--perform_validation", nargs="?", const=True, type=type_ensure_strtobool,
                        default=False,
                        help="If true, performs validation after training phase.")
    parser.add_argument("--augment_data_with_rotation", nargs="?", const=True, type=type_ensure_strtobool,
                        default=False,
                        help="If true, input data is augmented with synthetic rotational(90 degrees) input.")
    parser.add_argument("--augment_data_with_shadow", nargs="?", const=True, type=str,
                        default=None,
                        help="Given a method name, input data is augmented with shadow data(cycle_gan or simple")
    parser.add_argument("--augment_data_with_reflection", nargs="?", const=True, type=type_ensure_strtobool,
                        default=False,
                        help="If true, input data is augmented with synthetic reflection input.")
    parser.add_argument("--augmentation_random_threshold", nargs="?", type=float,
                        default=0.5,
                        help="Augmentation randomization threshold.")
    parser.add_argument("--offline_augmentation", nargs="?", const=True, type=type_ensure_strtobool,
                        default=False,
                        help="If added, input data is augmented offline in a randomized fashion.")
    parser.add_argument("--device", nargs="?", type=str,
                        default="gpu",
                        help="Device for processing: gpu, cpu or tpu")
    parser.add_argument("--test_ratio", nargs="?", type=float,
                        default=0.05,
                        help="Ratio of training data to use in testing")
    parser.add_argument("--train_ratio", nargs="?", type=float,
                        default=0.10,
                        help="Ratio of training data to use in validation, not accepted by all data set implementations")
    parser.add_argument("--split_count", nargs="?", type=int,
                        default=1,
                        help="Split count")
    parser.add_argument("--save_checkpoint_steps", nargs="?", type=int,
                        default=2000,
                        help="Save frequency of the checkpoint")
    parser.add_argument("--validation_steps", nargs="?", type=int,
                        default=40000,
                        help="Validation frequency")
    parser.add_argument("--max_evals", nargs="?", type=int,
                        default=2,
                        help="Maximum evaluation count for hyper parameter optimization")
    parser.add_argument("--algorithm_param_path", nargs="?", const=True, type=str,
                        default=None,
                        help="Algorithm parameter (json) data file path")
    parser.add_argument("--model_name", nargs="?", const=True, type=str,
                        default="CNNModelv2",
                        help="Model name to use in training")
    parser.add_argument("--all_data_shuffle_ratio", nargs="?", type=float,
                        default=None,
                        help="If given as a valid ratio, validation and training data is shuffled and redistributed")
    parser.add_argument("--log_model_params", nargs="?", const=True, type=type_ensure_strtobool,
                        default=False,
                        help="If added, logs model histogram to the tensorboard file.")


def add_parse_cmds_for_trainers(parser):
    parser.add_argument("--batch_size", nargs="?", type=int,
                        default=20,
                        help="Batch size")
    parser.add_argument("--step", nargs="?", const=True, type=int,
                        default=50000,
                        help="Step number to perform for training, either this parameter or epoch should be used")
    parser.add_argument("--epoch", nargs="?", const=True, type=int,
                        default=None,
                        help="Epoch number to traverse data, either this parameter or step should be used")


def add_parse_cmds_for_loggers(parser):
    parser.add_argument("--base_log_path", nargs="?", const=True, type=str,
                        default=os.path.dirname(__file__),
                        help="Base path for saving logs")
    parser.add_argument('--output_path', nargs='?', const=True, type=str,
                        default=os.path.dirname(__file__),
                        help='Path for saving output images')


def add_parse_cmds_for_loader(parser):
    parser.add_argument("--path", nargs="?", const=True, type=str,
                        default="/data/2013_DFTC/2013_DFTC",
                        help="Input data path")
    parser.add_argument("--loader_name", nargs="?", const=True, type=str,
                        default="GRSS2013DataLoader",
                        help="Data set loader name, values: GRSS2013DataLoader, GRSS2018DataLoader, "
                             "GULFPORTDataLoader, GULFPORTALTDataLoader, AVONDATALoader")
    parser.add_argument("--neighborhood", nargs="?", type=int,
                        default=0,
                        help="Neighborhood for data extraction, e.g. 1 means 3x3 patches")
