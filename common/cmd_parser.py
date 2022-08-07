import os
from distutils.util import strtobool


def type_ensure_strtobool(val):
    return bool(strtobool(str(val)))


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
                        default=os.getcwd(),
                        help="Base path for saving logs, default: working directory")
    parser.add_argument('--output_path', nargs='?', const=True, type=str,
                        default=os.getcwd(),
                        help="Path for saving output logs and images, default: working directory")


def add_parse_cmds_for_loaders(parser):
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
    parser.add_argument("--test_ratio", nargs="?", type=float,
                        default=0.05,
                        help="Ratio of training data to use in testing")
    parser.add_argument("--train_ratio", nargs="?", type=float,
                        default=0.10,
                        help="Ratio of training data to use in validation, not accepted by all data set impls.")


def add_parse_cmds_for_models(parser):
    parser.add_argument("--algorithm_param_path", nargs="?", const=True, type=str,
                        default=None,
                        help="Algorithm parameter (json) data file path")
    parser.add_argument("--model_name", nargs="?", const=True, type=str,
                        default="HYPELCNNModel",
                        help="Model to use in training, values: CAPModel, CONCNNModel, DUALCNNModel, HYPELCNNModel")


def add_parse_cmds_for_importers(parser):
    parser.add_argument("--importer_name", nargs="?", const=True, type=str,
                        default="InMemoryImporter",
                        help="Importer name, Values : GeneratorImporter, InMemoryImporter, TFRecordImporter")
