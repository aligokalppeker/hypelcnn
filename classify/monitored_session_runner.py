import gc

import tensorflow as tf
from tensorflow.python.training import summary_io
from tensorflow_core.python.training.basic_session_run_hooks import StopAtStepHook, NanTensorHook
from tf_slim import get_variables_to_restore, get_model_variables

from common.common_nn_ops import calculate_accuracy, TrainingResult, set_all_gpu_config, TextSummaryAtStartHook


def set_run_seed():
    # Set random seed as the same value to get consistent results
    tf.set_random_seed(1234)


def add_classification_summaries(cross_entropy, learning_rate, log_all_model_variables, testing_nn_params,
                                 validation_nn_params):
    tf.summary.scalar("training_cross_entropy", cross_entropy)
    tf.summary.scalar("training_learning_rate", learning_rate)
    tf.summary.text("test_confusion", tf.as_string(testing_nn_params.metrics.confusion))
    tf.summary.scalar("test_overall_accuracy", testing_nn_params.metrics.accuracy)
    tf.summary.text("validation_confusion", tf.as_string(validation_nn_params.metrics.confusion))
    tf.summary.scalar("validation_overall_accuracy", validation_nn_params.metrics.accuracy)
    tf.summary.scalar("validation_average_accuracy", validation_nn_params.metrics.mean_per_class_accuracy)
    tf.summary.scalar("validation_kappa", validation_nn_params.metrics.kappa)
    if log_all_model_variables:
        for variable in get_model_variables():
            tf.summary.histogram(variable.op.name, variable)


class InitHook(tf.train.SessionRunHook):

    def __init__(self, training_nn_params, training_tensor, augmentation_info, restorer, importer):
        self.importer = importer
        self.restorer = restorer
        self.augmentation_info = augmentation_info
        self.training_nn_params = training_nn_params
        self.training_tensor = training_tensor

    def after_create_session(self, session, coord):
        if self.augmentation_info.perform_shadow_augmentation:
            if self.augmentation_info.shadow_struct is not None and self.augmentation_info.shadow_struct.shadow_op_initializer is not None:
                self.augmentation_info.shadow_struct.shadow_op_initializer(self.restorer, session)

        self.importer.init_tensors(session, self.training_tensor, self.training_nn_params)


class ValidationHook(tf.train.SessionRunHook):

    def __init__(self, validation_nn_params, validation_tensor, class_range, required_steps, iteration, summary_dir,
                 importer):
        self.importer = importer
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
        self._writer = summary_io.SummaryWriterCache.get(self.summary_dir)

    def after_run(self, run_context, run_values):
        session = run_context.session
        iteration = session.run(self._global_step_tensor)
        if self.validation_nn_params is not None:
            if (iteration == self.required_steps - 1) or (iteration % self._iteration_count == 1 and iteration != 1):
                self.importer.init_tensors(session, self.validation_tensor,
                                           self.validation_nn_params)
                self.validation_accuracy, class_recall, class_precisions, kappa, mean_per_class_accuracy = calculate_accuracy(
                    session, self.validation_nn_params, self.class_range)

                print('Validation metrics #%d : Overall accuracy=%g, Class based average accuracy=%g, Kappa=%g' % (
                    iteration, self.validation_accuracy, mean_per_class_accuracy, kappa))

                # print('Class based precision=', array2string(class_precisions, precision=2), mean(class_precisions))
                # print('Class based recall=', array2string(class_recall, precision=2), mean(class_recall))

                # Log all the results to tf summary
                self._writer.add_summary(session.run(tf.get_collection("summary_op")[0]), iteration)
                # Collect unnecessary data
                gc.collect()


class TestHook(tf.train.SessionRunHook):

    def __init__(self, testing_nn_params, testing_tenser, cross_entropy, test_iteration_count, class_range, importer):
        self.importer = importer
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
            self.importer.init_tensors(session, self.testing_tensor, self.testing_nn_params)
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
                          importer, flags_as_json_str, alg_params_as_json_str):
    read_op_value = None
    augmentation_restorer = None
    if augmentation_info.perform_shadow_augmentation:
        if augmentation_info.shadow_struct is not None and augmentation_info.shadow_struct.shadow_op_initializer is not None:
            augmentation_restorer = augmentation_info.shadow_struct.shadow_op_creater()
            # Ready ops are overriden, as default ready ops awaits all variables to be initialized
            # but actually some variables(such as cycle-gan graphs) are not initialized but restored
            read_op_value = tf.constant([])

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    set_all_gpu_config()
    master = ""

    validation_hook = ValidationHook(validation_nn_params, validation_tensor, class_range, required_steps,
                                     validation_steps,
                                     log_dir, importer)
    test_iteration_count = 100
    test_hook = TestHook(testing_nn_params, testing_tensor, cross_entropy, test_iteration_count, class_range, importer)
    initializer_hook = InitHook(training_nn_params, training_tensor, augmentation_info, augmentation_restorer, importer)
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

    # Only restore nn core variables along with the optimizer and global step variables
    nn_core_restorer = tf.train.Saver(
        max_to_keep=20,
        var_list=get_variables_to_restore(include=["nn_core"]) +
                 get_variables_to_restore(include=["global_step"]) +
                 get_variables_to_restore(include=["training_optimizer"]), name="nn_core_restorer")
    training_scaffold = tf.train.Scaffold(saver=nn_core_restorer,
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

    result = TrainingResult(validation_accuracy=validation_hook.validation_accuracy,
                            test_accuracy=test_hook.testing_accuracy, loss=test_hook.loss)
    return result
