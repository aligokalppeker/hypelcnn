import glob
import os
import sys
from pathlib import Path

import numpy
import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError

from stat_extractor import extract_statistics_info, print_statistics_info


def main():
    event_paths = None
    filtered_steps = []

    for index, filename in enumerate(sys.argv):
        if index == 1:
            absolute_path = os.path.join(os.path.dirname(__file__), sys.argv[index])
            event_paths = glob.glob(os.path.join(absolute_path, "event*"))
        elif index > 1:
            filtered_steps.append(int(sys.argv[index]))

    confusion_matrix_list = []
    for event_path in event_paths:
        parent_dir = Path(event_path).parent
        try:
            events = tf.train.summary_iterator(event_path)
            for e in events:
                # print(e.step)
                if not filtered_steps or e.step in filtered_steps:
                    for val in e.summary.value:
                        # print(val.tag + ":" + str(val.simple_value))
                        if val.tag == "validation_confusion":
                            print("Step %i in %s" % (e.step, event_path))
                            width = val.tensor.tensor_shape.dim[0].size
                            height = val.tensor.tensor_shape.dim[1].size
                            confusion_matrix = numpy.zeros([width, height], dtype=int)
                            for width_index in range(0, width):
                                for height_index in range(0, height):
                                    confusion_matrix[height_index][width_index] = \
                                        int(val.tensor.string_val[(width * height_index) + width_index])
                            record_filename = \
                                parent_dir.parent.name + "_" + parent_dir.name + "_s" + str(e.step) + ".csv"
                            full_record_path = os.path.join(".", record_filename)
                            print("Saving to file:", full_record_path)
                            numpy.savetxt(full_record_path, confusion_matrix, fmt="%d", delimiter=",")
                            confusion_matrix_list.append(confusion_matrix)
        except DataLossError:
            print("Error reading summary file: ", event_path)
            pass
    print_statistics_info(extract_statistics_info(confusion_matrix_list))


if __name__ == '__main__':
    main()
