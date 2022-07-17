import gc
import os

import tensorflow as tf
from tensorflow import constant
from tensorflow.contrib import slim
from tensorflow.contrib import tpu
from tensorflow.contrib.learn.python.learn.summary_writer_cache import SummaryWriterCache
from tensorflow.python.framework import ops
from tensorflow.python.training.basic_session_run_hooks import StopAtStepHook, NanTensorHook
from tensorflow.python.training.monitored_session import Scaffold

from common_nn_operations import calculate_accuracy, TrainingResult, set_all_gpu_config, TextSummaryAtStartHook


class InitializerHook(tf.train.SessionRunHook):

    def __init__(self, training_nn_params, training_tensor, augmentation_info, restorer):
        self.restorer = restorer
        self.augmentation_info = augmentation_info
        self.training_nn_params = training_nn_params
        self.training_tensor = training_tensor

    def after_create_session(self, session, coord):
        if self.augmentation_info.perform_shadow_augmentation:
            if self.augmentation_info.shadow_struct is not None and self.augmentation_info.shadow_struct.shadow_op_initializer is not None:
                self.augmentation_info.shadow_struct.shadow_op_initializer(self.restorer, session)

        self.training_tensor.importer.perform_tensor_initialize(session, self.training_tensor, self.training_nn_params)


class ValidationHook(tf.train.SessionRunHook):

    def __init__(self, validation_nn_params, validation_tensor, class_range, required_steps, iteration, summary_dir):
        self.required_steps = required_steps
        self.validation_nn_params = validation_nn_params
        self.validation_tensor = validation_tensor
        self.validation_accuracy = 0
        self.class_range = class_range
        self.summary_dir = summary_dir
        self._iteration_count = iteration
        self._global_step_tensor = None
        self._writer = None

    def after_create_session(self, session, coord):
        self._global_step_tensor = tf.train.get_global_step()
        self._writer = SummaryWriterCache.get(self.summary_dir)

    def after_run(self, run_context, run_values):
        session = run_context.session
        iteration = session.run(self._global_step_tensor)
        if self.validation_nn_params is not None:
            if (iteration == self.required_steps - 1) or (iteration % self._iteration_count == 1 and iteration != 1):
                self.validation_tensor.importer.perform_tensor_initialize(session, self.validation_tensor,
                                                                          self.validation_nn_params)
                self.validation_accuracy, class_recall, class_precisions, kappa, mean_per_class_accuracy = calculate_accuracy(
                    session, self.validation_nn_params, self.class_range)

                print('Validation metrics #%d : Overall accuracy=%g, Class based average accuracy=%g, Kappa=%g' % (
                    iteration, self.validation_accuracy, mean_per_class_accuracy, kappa))

                # print('Class based precision=', array2string(class_precisions, precision=2), mean(class_precisions))
                # print('Class based recall=', array2string(class_recall, precision=2), mean(class_recall))

                # Log all the results to tf summary
                summary_op = ops.get_collection(ops.GraphKeys.SUMMARY_OP)
                self._writer.add_summary(session.run(summary_op[0]), iteration)
                # Collect unnecessary data
                gc.collect()


class TestHook(tf.train.SessionRunHook):

    def __init__(self, testing_nn_params, testing_tenser, cross_entropy, test_iteration_count, class_range):
        self.testing_nn_params = testing_nn_params
        self.testing_tensor = testing_tenser
        self.testing_accuracy = 0
        self.loss = 0
        self.cross_entropy = cross_entropy
        self._global_step_tensor = None
        self._test_iteration_count = test_iteration_count
        self.class_range = class_range

    def after_create_session(self, session, coord):
        self._global_step_tensor = tf.train.get_global_step()

    def after_run(self, run_context, run_values):
        session = run_context.session
        iteration = session.run(self._global_step_tensor)
        if iteration % self._test_iteration_count == 1:
            self.__perform_action(session, iteration)

    def end(self, session):
        self.__perform_action(session, session.run(self._global_step_tensor))

    def __perform_action(self, session, iteration):
        self.loss = session.run(self.cross_entropy)
        # Perform test if there is test data
        if self.testing_nn_params.data_with_labels.data.size != 0:
            self.testing_tensor.importer.perform_tensor_initialize(session, self.testing_tensor, self.testing_nn_params)
            self.testing_accuracy, class_recall, class_precisions, kappa, mean_per_class_accuracy = calculate_accuracy(
                session, self.testing_nn_params, self.class_range)
        print('Training step=%d, Testing accuracy=%g, loss=%.5f' % (iteration, self.testing_accuracy, self.loss))


def run_monitored_session(cross_entropy, log_dir, class_range,
                          save_checkpoint_steps, validation_steps,
                          train_step, required_steps,
                          augmentation_info, device,
                          training_nn_params, training_tensor,
                          testing_nn_params, testing_tensor,
                          validation_nn_params, validation_tensor,
                          flags_as_json_str, alg_params_as_json_str):
    read_op_value = None
    augmentation_restorer = None
    if augmentation_info.perform_shadow_augmentation:
        if augmentation_info.shadow_struct is not None and augmentation_info.shadow_struct.shadow_op_initializer is not None:
            augmentation_restorer = augmentation_info.shadow_struct.shadow_op_creater()
            # Ready ops are overriden, as default ready ops awaits all variables to be initialized
            # but actually some of the variables(such as cycle-gan graphs) are not initialized but restored
            read_op_value = constant([])

    is_gpu_or_cpu = (device == "gpu" or device == "cpu")
    if is_gpu_or_cpu:
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        set_all_gpu_config()
        master = ''
    else:
        config = None
        tpu_worker = "grpc://" + os.environ["COLAB_TPU_ADDR"]
        # master = TPUClusterResolver(tpu=tpu_worker).get_master()
        master = tpu_worker
        print("TPU master")
        print(master)

    validation_hook = ValidationHook(validation_nn_params, validation_tensor, class_range, required_steps,
                                     validation_steps,
                                     log_dir)
    test_iteration_count = 100
    test_hook = TestHook(testing_nn_params, testing_tensor, cross_entropy, test_iteration_count, class_range)
    initializer_hook = InitializerHook(training_nn_params, training_tensor, augmentation_info, augmentation_restorer)
    stop_on_step_hook = StopAtStepHook(last_step=required_steps - 1)
    nan_tensor_hook = NanTensorHook(loss_tensor=cross_entropy, fail_on_nan_loss=False)
    flags_log_hook = TextSummaryAtStartHook(log_dir=log_dir, name="flags", value=flags_as_json_str)
    algparams_log_hook = TextSummaryAtStartHook(log_dir=log_dir, name="algorithm_params", value=alg_params_as_json_str)

    hooks = [initializer_hook,
             validation_hook,
             test_hook,
             stop_on_step_hook,
             nan_tensor_hook,
             flags_log_hook,
             algparams_log_hook]

    if is_gpu_or_cpu:
        # Only restore nn core variables along with the optimizer and global step variables
        nn_core_restorer = tf.train.Saver(
            max_to_keep=20,
            var_list=slim.get_variables_to_restore(include=["nn_core"]) +
                     slim.get_variables_to_restore(include=["global_step"]) +
                     slim.get_variables_to_restore(include=["training_optimizer"]), name="nn_core_restorer")
        training_scaffold = Scaffold(saver=nn_core_restorer,
                                     ready_for_local_init_op=read_op_value,
                                     ready_op=read_op_value)

        session = tf.train.MonitoredTrainingSession(master=master,
                                                    checkpoint_dir=log_dir,
                                                    summary_dir=log_dir,
                                                    config=config, is_chief=True,
                                                    save_summaries_steps=test_iteration_count,
                                                    save_checkpoint_steps=save_checkpoint_steps,
                                                    scaffold=training_scaffold,
                                                    hooks=hooks)
        # session = LocalCLIDebugWrapperSession(session)
        with session as monitored_sess:
            while not monitored_sess.should_stop():
                monitored_sess.run([train_step])
    else:
        session = tf.Session(target=master, config=config)
        session.run(tpu.initialize_system())
        session.run(tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer()))
        initializer_hook.after_create_session(session, None)
        step_tensor = tf.train.get_global_step()
        while session.run(step_tensor) < required_steps:
            try:
                session.run(train_step)
                test_hook.after_run_with_session(session)
            except tf.errors.OutOfRangeError:
                break

        validation_hook.end(session)
        session.run(tpu.shutdown_system())
        session.close()

    result = TrainingResult(validation_accuracy=validation_hook.validation_accuracy,
                            test_accuracy=test_hook.testing_accuracy, loss=test_hook.loss)
    return result
