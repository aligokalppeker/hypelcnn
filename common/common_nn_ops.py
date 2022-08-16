import numpy
import tensorflow as tf
from numba import jit
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.metrics_impl import metric_variable
from tensorflow.python.training import summary_io
from tensorflow_core.contrib.metrics.python.ops.metric_ops import cohen_kappa
from tensorflow_core.python.data.experimental import shuffle_and_repeat, prefetch_to_device
from tensorflow_core.python.training.session_run_hook import SessionRunHook
from tf_slim.learning import create_train_op
from tifffile import imread
from tqdm import tqdm

from common.common_ops import get_class, is_integer_num

INVALID_TARGET_VALUE = 255


class DataSet:
    def __init__(self, shadow_creator_dict, casi, lidar, neighborhood, normalize,
                 casi_min=None, casi_max=None, lidar_min=None, lidar_max=None) -> None:
        self.neighborhood = neighborhood
        self.lidar = lidar
        self.casi = casi
        self.casi_unnormalized_dtype = self.casi.dtype
        self.shadow_creator_dict = shadow_creator_dict

        # Padding part
        pad_size = ((self.neighborhood, self.neighborhood), (self.neighborhood, self.neighborhood), (0, 0))
        if self.lidar is not None:
            self.lidar = numpy.pad(self.lidar, pad_size, mode="symmetric")

        if self.casi is not None:
            self.casi = numpy.pad(self.casi, pad_size, mode="symmetric")

        # Normalization
        self.casi_min = 0
        self.casi_max = 1
        self.lidar_min = 0
        self.lidar_max = 1
        if normalize:
            if self.lidar is not None:
                self.lidar_min = numpy.min(self.lidar) if lidar_min is None else lidar_min
                self.lidar -= self.lidar_min
                self.lidar_max = numpy.max(self.lidar) if lidar_max is None else lidar_max
                self.lidar = self.lidar / self.lidar_max

            if self.casi is not None:
                self.casi_min = numpy.min(self.casi, axis=(0, 1)) if casi_min is None else casi_min
                self.casi -= self.casi_min
                self.casi_max = numpy.max(self.casi, axis=(0, 1)) if casi_max is None else casi_max
                self.casi = self.casi / self.casi_max.astype(numpy.float32)

    def get_data_shape(self):
        dim = self.neighborhood * 2 + 1
        channel_count = self.casi.shape[2]
        if self.lidar is not None:
            channel_count = channel_count + 1
        return [dim, dim, channel_count]

    def get_casi_band_count(self):
        return self.casi.shape[2]

    def get_scene_shape(self):
        padding = self.neighborhood * 2
        primary_shape_to_consider = self.lidar
        if primary_shape_to_consider is None:
            primary_shape_to_consider = self.casi
        return [primary_shape_to_consider.shape[0] - padding,
                primary_shape_to_consider.shape[1] - padding]

    def get_unnormalized_casi_dtype(self):
        return self.casi_unnormalized_dtype


class NNParams:
    def __init__(self, input_iterator, data_with_labels, metrics, predict_tensor):
        self.predict_tensor = predict_tensor
        self.metrics = metrics
        self.data_with_labels = data_with_labels
        self.input_iterator = input_iterator


class ModelInputParams:
    def __init__(self, x, y, device_id, is_training):
        self.is_training = is_training
        self.device_id = device_id
        self.y = y
        self.x = x


class HistogramTensorPair:
    def __init__(self, tensor, name):
        self.name = name
        self.tensor = tensor


class ModelOutputTensors:
    def __init__(self, y_conv, image_output, image_original, histogram_tensors):
        self.image_original = image_original
        self.image_output = image_output
        self.y_conv = y_conv
        self.histogram_tensors = histogram_tensors


class TrainingResult:
    def __init__(self, validation_accuracy, test_accuracy, loss):
        self.loss = loss
        self.test_accuracy = test_accuracy
        self.validation_accuracy = validation_accuracy


class MetricOpsHolder:
    def __init__(self, combined_metric_update_op, metric_variables_reset_op, accuracy, mean_per_class_accuracy, kappa,
                 confusion):
        self.kappa = kappa
        self.mean_per_class_accuracy = mean_per_class_accuracy
        self.metric_variables_reset_op = metric_variables_reset_op
        self.confusion = confusion
        self.accuracy = accuracy
        self.combined_metric_update_op = combined_metric_update_op


class AugmentationInfo:
    def __init__(self, shadow_struct, perform_shadow_augmentation, perform_rotation_augmentation,
                 perform_reflection_augmentation, offline_or_online, augmentation_random_threshold):
        self.offline_or_online = offline_or_online
        self.perform_reflection_augmentation = perform_reflection_augmentation
        self.perform_rotation_augmentation = perform_rotation_augmentation
        self.perform_shadow_augmentation = perform_shadow_augmentation
        self.shadow_struct = shadow_struct
        self.augmentation_random_threshold = augmentation_random_threshold


@jit(nopython=True)
def get_data_point_func(casi, lidar, neighborhood, point):
    start_x = point[0]  # + neighborhood(pad offset) - neighborhood(back step); padding and back shift makes delta zero
    start_y = point[1]  # + neighborhood(pad offset) - neighborhood(back step); padding and back shift makes delta zero
    end_x = start_x + (2 * neighborhood) + 1
    end_y = start_y + (2 * neighborhood) + 1
    value = numpy.concatenate(
        (casi[start_y:end_y:1, start_x:end_x:1, :], lidar[start_y:end_y:1, start_x:end_x:1, :]), axis=2)
    return value


def training_nn_iterator(data_set, augmentation_info, batch_size, num_epochs, device, prefetch_size):
    main_cycle_data_set = data_set.apply(shuffle_and_repeat(buffer_size=10000, count=num_epochs))

    if augmentation_info.offline_or_online is False:
        main_cycle_data_set = add_augmentation_graph(main_cycle_data_set, augmentation_info,
                                                     perform_rotation_augmentation_random,
                                                     perform_shadow_augmentation_random,
                                                     perform_reflection_augmentation_random)

    main_cycle_data_set = main_cycle_data_set.batch(batch_size)
    main_cycle_data_set = main_cycle_data_set.prefetch(prefetch_size)

    if augmentation_info.offline_or_online is True:
        main_cycle_data_set = add_augmentation_graph(main_cycle_data_set, augmentation_info,
                                                     perform_rotation_augmentation,
                                                     perform_shadow_augmentation,
                                                     perform_reflection_augmentation)
    main_cycle_data_set = main_cycle_data_set.apply(prefetch_to_device(device, 10000))
    return tf.compat.v1.data.make_initializable_iterator(main_cycle_data_set)


def simple_nn_iterator(data_set, batch_size):
    return tf.compat.v1.data.make_initializable_iterator(data_set.batch(batch_size).prefetch(10000))


def optimize_nn(deep_nn_template, images, labels, device_id, name_prefix, algorithm_params, loss_func):
    tensor_outputs = deep_nn_template(
        model_input_params=ModelInputParams(x=images, y=labels, device_id=device_id, is_training=True),
        algorithm_params=algorithm_params)

    with tf.compat.v1.name_scope(name_prefix + '_loss'):
        cross_entropy = tf.reduce_mean(input_tensor=loss_func(tensor_outputs, labels))
    with tf.compat.v1.name_scope(name_prefix + '_optimizer'):
        global_step = tf.compat.v1.train.get_or_create_global_step()
        learning_rate = tf.compat.v1.train.exponential_decay(algorithm_params["learning_rate"],
                                                             global_step,
                                                             algorithm_params["learning_rate_decay_step"],
                                                             algorithm_params["learning_rate_decay_factor"],
                                                             staircase=True)

        if isinstance(algorithm_params["optimizer"], tuple) or isinstance(algorithm_params["optimizer"], list):
            if algorithm_params["optimizer"][0] == "MomentumOptimizer":
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate,
                                                                 momentum=algorithm_params["optimizer"][1],
                                                                 name="nn_core/Momentum")
        else:
            if algorithm_params["optimizer"] == "AdamOptimizer":
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate, name="nn_core/Adam")

        train_step = create_train_op(cross_entropy, optimizer, global_step=global_step)

        # This part is required for batch normalization to work
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

    return tensor_outputs.y_conv, cross_entropy, learning_rate, train_step


def create_metric_tensors(labels, y_conv, class_range, name_prefix):
    num_classes = class_range.stop
    with tf.compat.v1.name_scope(name_prefix + "_metrics"):
        prediction = tf.argmax(input=y_conv, axis=1)
        label = tf.argmax(input=labels, axis=1)

        # the streaming accuracy (lookup and update tensors)
        accuracy, accuracy_update = tf.compat.v1.metrics.accuracy(
            label, prediction, name='accuracy')
        mean_per_class_accuracy, mean_per_class_accuracy_update = tf.compat.v1.metrics.mean_per_class_accuracy(
            label, prediction, num_classes, name='mean_per_class_accuracy')
        kappa, kappa_update = cohen_kappa(label, prediction, num_classes, name='kappa')
        # Compute a per-batch confusion
        batch_confusion = tf.math.confusion_matrix(labels=label, predictions=prediction,
                                                   num_classes=num_classes,
                                                   name='batch_confusion')
        # Create an accumulator variable to hold the counts
        confusion_var = metric_variable([num_classes, num_classes], dtype=tf.int32, name='confusion')
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion_var.assign(confusion_var + batch_confusion)

        metric_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
                                                       scope=name_prefix + "_metrics")
        metric_variables_reset_op = tf.compat.v1.variables_initializer(var_list=metric_variables)

        # Combine streaming accuracy and confusion matrix updates in one op
        combined_metric_update_op = tf.group(accuracy_update, mean_per_class_accuracy_update,
                                             kappa_update, confusion_update)

    return MetricOpsHolder(combined_metric_update_op=combined_metric_update_op,
                           metric_variables_reset_op=metric_variables_reset_op,
                           accuracy=accuracy,
                           confusion=confusion_var,
                           mean_per_class_accuracy=mean_per_class_accuracy,
                           kappa=kappa)


def calculate_class_accuracies_using_confusion(confusion_matrix, class_range):
    class_space = class_range.stop
    class_precisions = numpy.zeros(class_space)
    class_recall = numpy.zeros(class_space)
    for index in class_range:
        total_ground_truths = numpy.sum(confusion_matrix[index, :])
        if total_ground_truths != 0:
            class_recall[index] = confusion_matrix[index, index] / total_ground_truths
        total_predictions = numpy.sum(confusion_matrix[:, index])
        if total_predictions != 0:
            class_precisions[index] = confusion_matrix[index, index] / total_predictions

    return class_recall[class_range], class_precisions[class_range]


def calculate_accuracy(sess, nn_params, class_range):
    sess.run(nn_params.metrics.metric_variables_reset_op)

    while True:
        try:
            # Collect data partially
            sess.run(nn_params.metrics.combined_metric_update_op)
        except tf.errors.OutOfRangeError:
            # Calculate metrics when data collection is completed
            confusion_matrix = sess.run(nn_params.metrics.confusion)
            overall_accuracy = sess.run(nn_params.metrics.accuracy)
            kappa = sess.run(nn_params.metrics.kappa)
            mean_per_class_accuracy = sess.run(nn_params.metrics.mean_per_class_accuracy)
            break
    class_recall, class_precisions = calculate_class_accuracies_using_confusion(confusion_matrix, class_range)
    return overall_accuracy, class_recall, class_precisions, kappa, mean_per_class_accuracy


def perform_prediction(sess, nn_params, prediction_result_arr):
    current_prediction_index = 0
    progress_bar = tqdm(total=1)
    while True:
        try:
            current_prediction = sess.run(tf.argmax(input=nn_params.predict_tensor, axis=1))
            next_prediction_index = current_prediction_index + current_prediction.shape[0]
            prediction_result_arr[current_prediction_index:next_prediction_index] = current_prediction.astype(
                numpy.uint8)
            progress_bar.update((next_prediction_index - current_prediction_index) / prediction_result_arr.shape[0])
            current_prediction_index = next_prediction_index
        except tf.errors.OutOfRangeError:
            progress_bar.close()
            break


def create_graph(training_data_set, testing_data_set, validation_data_set, class_range,
                 batch_size, prefetch_size, device_id, num_epochs, algorithm_params, model,
                 augmentation_info, create_separate_validation_branch):
    deep_nn_template = tf.compat.v1.make_template("nn_core", model.create_tensor_graph, class_count=class_range.stop)
    ####################################################################################
    training_input_iter = training_nn_iterator(training_data_set, augmentation_info, batch_size, num_epochs, device_id,
                                               prefetch_size)
    images, labels = training_input_iter.get_next()

    training_y_conv, cross_entropy, learning_rate, train_step = optimize_nn(deep_nn_template,
                                                                            images, labels,
                                                                            device_id=device_id,
                                                                            name_prefix="training",
                                                                            algorithm_params=algorithm_params,
                                                                            loss_func=model.get_loss_func)

    train_nn_params = NNParams(input_iterator=training_input_iter, data_with_labels=None, metrics=None,
                               predict_tensor=None)
    ####################################################################################
    testing_input_iter = simple_nn_iterator(testing_data_set, batch_size)
    testing_images, testing_labels = testing_input_iter.get_next()
    model_input_params = ModelInputParams(x=testing_images, y=None, device_id=device_id, is_training=False)
    testing_tensor_outputs = deep_nn_template(model_input_params, algorithm_params=algorithm_params)
    test_metric_ops_holder = create_metric_tensors(testing_labels, testing_tensor_outputs.y_conv, class_range,
                                                   "testing")
    testing_nn_params = NNParams(input_iterator=testing_input_iter, data_with_labels=None,
                                 metrics=test_metric_ops_holder, predict_tensor=None)
    ####################################################################################
    validation_nn_params = NNParams(input_iterator=testing_input_iter, data_with_labels=None,
                                    metrics=test_metric_ops_holder, predict_tensor=None)
    if create_separate_validation_branch:
        validation_input_iter = simple_nn_iterator(validation_data_set, batch_size)
        validation_images, validation_labels = validation_input_iter.get_next()
        validation_model_input_params = ModelInputParams(x=validation_images, y=None, device_id=device_id,
                                                         is_training=False)
        validation_tensor_outputs = deep_nn_template(validation_model_input_params, algorithm_params=algorithm_params)
        validation_metric_ops_holder = create_metric_tensors(validation_labels, validation_tensor_outputs.y_conv,
                                                             class_range,
                                                             "validation")
        validation_nn_params = NNParams(input_iterator=validation_input_iter, data_with_labels=None,
                                        metrics=validation_metric_ops_holder, predict_tensor=None)
    ####################################################################################

    return cross_entropy, learning_rate, testing_nn_params, train_nn_params, validation_nn_params, train_step


def add_augmentation_graph(main_cycle_dataset, augmentation_info, rotation_method, shadow_method, flip_method):
    if augmentation_info.perform_rotation_augmentation:
        main_cycle_dataset = main_cycle_dataset.map(
            lambda param_x, param_y_: rotation_method(param_x, param_y_, augmentation_info),
            num_parallel_calls=4)
    if augmentation_info.perform_shadow_augmentation:
        main_cycle_dataset = main_cycle_dataset.map(
            lambda param_x, param_y_: shadow_method(param_x, param_y_, augmentation_info),
            num_parallel_calls=4)
    if augmentation_info.perform_reflection_augmentation:
        main_cycle_dataset = main_cycle_dataset.map(
            lambda param_x, param_y_: flip_method(param_x, param_y_, augmentation_info),
            num_parallel_calls=4)
    return main_cycle_dataset


def perform_rotation_augmentation(images, labels, augmentation_info):
    with tf.compat.v1.name_scope("rotation_augmentation"):
        transforms = [images]
        label_transforms = [labels]

        for index in range(1, 4):
            transforms.append(tf.image.rot90(images, index))
            label_transforms.append(labels)

        images = tf.concat(transforms, axis=0)
        labels = tf.concat(label_transforms, axis=0)

    return images, labels


def perform_shadow_augmentation(images, labels, augmentation_info):
    shadow_op = augmentation_info.shadow_struct.shadow_op
    with tf.compat.v1.name_scope("shadow_augmentation"):
        transforms = [images]
        label_transforms = [labels]

        transforms.append(shadow_op(images))
        label_transforms.append(labels)

        images = tf.concat(transforms, axis=0)
        labels = tf.concat(label_transforms, axis=0)

    return images, labels


def perform_reflection_augmentation(images, labels, augmentation_info):
    with tf.compat.v1.name_scope("reflection_augmentation"):
        transforms = [images]
        label_transforms = [labels]

        transforms.append(tf.image.flip_left_right(images))
        label_transforms.append(labels)

        transforms.append(tf.image.flip_up_down(images))
        label_transforms.append(labels)

        images = tf.concat(transforms, axis=0)
        labels = tf.concat(label_transforms, axis=0)

    return images, labels


def perform_rotation_augmentation_random(images, labels, augmentation_info):
    with tf.compat.v1.name_scope("rotation_augmentation"):
        with tf.device("/cpu:0"):
            # shp = tf.shape(images)
            # batch_size = shp[0]
            angles = tf.random.uniform([1], 0, 3, dtype="int32")
            images = tf.image.rot90(images, angles[0])

    return images, labels


def perform_shadow_augmentation_random(images, labels, augmentation_info):
    if augmentation_info.shadow_struct is not None:
        shadow_op = augmentation_info.shadow_struct.shadow_op
        with tf.compat.v1.name_scope('shadow_augmentation'):
            with tf.device('/cpu:0'):
                rand_number = tf.random.uniform([1], 0, 1.0)[0]
                images = tf.cond(pred=tf.less(rand_number, augmentation_info.augmentation_random_threshold),
                                 true_fn=lambda: shadow_op(images),
                                 false_fn=lambda: images)
    return images, labels


def perform_reflection_augmentation_random(images, labels, augmentation_info):
    with tf.compat.v1.name_scope("reflection_augmentation"):
        with tf.device("/cpu:0"):
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)

    return images, labels


def get_model_from_name(model_name):
    return get_class("nnmodel." + model_name + '.' + model_name)()


def get_importer_from_name(importer_name):
    return get_class("importer." + importer_name + '.' + importer_name)()


def get_loader_from_name(loader_name, path):
    return get_class("loader." + loader_name + '.' + loader_name)(path)


def create_colored_image(target_image, color_list):
    image_colorized = numpy.zeros([target_image.shape[0], target_image.shape[1], 3], dtype=numpy.uint8)
    for col_index in range(0, target_image.shape[0]):
        for row_index in range(0, target_image.shape[1]):
            target_value = target_image[col_index, row_index]
            if target_value < len(color_list):
                image_colorized[col_index, row_index] = color_list[target_value]
    return image_colorized


def create_target_image_via_samples(sample_set, scene_shape):
    image = numpy.full([scene_shape[0], scene_shape[1]], INVALID_TARGET_VALUE, dtype=numpy.uint8)
    targets = numpy.vstack([sample_set.training_targets, sample_set.test_targets, sample_set.validation_targets])
    for point in targets.astype(int):
        image[point[1], point[0]] = point[2]
    return image


def calculate_shadow_ratio(casi, shadow_map, shadow_map_inverse):
    shadow_map_multi = \
        numpy.repeat(numpy.expand_dims(shadow_map == 0, axis=2), repeats=casi.shape[2], axis=2)
    shadow_map_inverse_multi = \
        numpy.repeat(numpy.expand_dims(shadow_map_inverse == 0, axis=2), repeats=casi.shape[2], axis=2)

    data_in_shadow_map = numpy.ma.array(data=casi, mask=shadow_map_multi)
    data_in_non_shadow_map = numpy.ma.array(data=casi, mask=shadow_map_inverse_multi)

    ratio_per_band = data_in_non_shadow_map.mean(axis=(0, 1)) / data_in_shadow_map.mean(axis=(0, 1))
    return ratio_per_band.filled().astype(numpy.float32)


def read_targets_from_image(targets, class_range):
    result = numpy.array([], dtype=int).reshape(0, 3)
    for target_index in class_range:
        target_locations = numpy.where(targets == target_index)
        target_locations_as_array = numpy.transpose(
            numpy.vstack((target_locations[1].astype(int), target_locations[0].astype(int))))
        target_index_as_array = numpy.full((len(target_locations_as_array), 1), target_index)
        result = numpy.vstack([result, numpy.hstack((target_locations_as_array, target_index_as_array))])
    return result


def shuffle_training_data_using_ratio(result, train_data_ratio):
    validation_set = None
    train_set = None
    shuffler = StratifiedShuffleSplit(n_splits=1, train_size=train_data_ratio)
    for train_index, test_index in shuffler.split(result[:, 0:1], result[:, 2]):
        validation_set = result[test_index]
        train_set = result[train_index]
    return train_set, validation_set


def shuffle_training_data_using_size(class_count, result, train_data_size, validation_size):
    sample_id_list = result[:, 2]
    train_set = numpy.empty([0, result.shape[1]], dtype=numpy.int)
    validation_set = numpy.empty([0, result.shape[1]], dtype=numpy.int)
    for sample_class in class_count:
        id_for_class = numpy.where(sample_id_list == sample_class)[0]
        class_sample_count = id_for_class.shape[0]
        if class_sample_count > 0:
            all_index = numpy.arange(class_sample_count)

            if class_sample_count < train_data_size:
                # Select 0.90 of elements in case of overflow
                train_index = numpy.random.choice(class_sample_count, (class_sample_count * 9) // 10, replace=False)
            else:
                train_index = numpy.random.choice(class_sample_count, train_data_size, replace=False)

            validation_index = numpy.array([index for index in all_index if index not in train_index])
            if validation_size is not None:
                validation_index_size = validation_index.shape[0]
                validation_size = min(validation_size, validation_index_size)
                rand_indices = numpy.random.choice(validation_index_size, validation_size, replace=False)
                validation_index = validation_index[rand_indices]
            # add elements
            train_set = numpy.vstack([train_set, result[id_for_class[train_index], :]])
            validation_set = numpy.vstack([validation_set, result[id_for_class[validation_index], :]])
    return train_set, validation_set


def shuffle_test_data_using_ratio(train_set, test_data_ratio):
    # Empty set for 0 ratio for testing
    test_set = numpy.empty([0, train_set.shape[1]])
    if test_data_ratio > 0:
        train_shuffler = StratifiedShuffleSplit(n_splits=1, test_size=test_data_ratio, random_state=0)
        for train_index, test_index in train_shuffler.split(train_set[:, 0:1], train_set[:, 2]):
            test_set = train_set[test_index]
            train_set = train_set[train_index]
    return test_set, train_set


def scale_in_to_out(input_data, output_data, axis_no):
    input_channel_size = input_data.get_shape()[axis_no].value
    output_channel_size = output_data.get_shape()[axis_no].value
    scale_ratio = input_channel_size / output_channel_size
    inv_scale_ratio = 1 / scale_ratio

    if is_integer_num(inv_scale_ratio):
        if int(inv_scale_ratio) == 1:
            result = input_data
        else:
            result = tf.repeat(input=input_data, axis=axis_no, repeats=int(inv_scale_ratio))
    else:
        output_data_indice_list = []
        for output_data_index in range(0, output_channel_size):
            target_index_no = min(round(output_data_index * scale_ratio), input_channel_size - 1)
            output_data_indice_list.append(target_index_no)

        result = tf.gather(input_data, output_data_indice_list, axis=axis_no)
    return result


def load_shadow_map_common(data_set, neighborhood, shadow_file_name):
    shadow_map = numpy.pad(imread(shadow_file_name), neighborhood, mode="symmetric")
    shadow_ratio = None if data_set is None else calculate_shadow_ratio(data_set.casi, shadow_map,
                                                                        numpy.logical_not(shadow_map).astype(int))
    return shadow_map, shadow_ratio


def set_all_gpu_config():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


class TextSummaryAtStartHook(SessionRunHook):

    def __init__(self, log_dir, name, value):
        super().__init__()
        self._log_dir = log_dir
        value = "<pre>" + value + "</pre>"
        self._summary_text_tensor = tf.compat.v1.summary.text(name, tf.constant(value=value, name=name + "_tensor"),
                                                              collections=["custom"])

    def after_create_session(self, session, coord):
        current_iteration = session.run(tf.compat.v1.train.get_global_step())
        summary_io.SummaryWriterCache.get(self._log_dir).add_summary(session.run(self._summary_text_tensor),
                                                                     current_iteration)
