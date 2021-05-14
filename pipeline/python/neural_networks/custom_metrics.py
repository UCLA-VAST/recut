import tensorflow as tf
from tensorflow.keras.metrics import Metric, MeanIoU, Precision, Recall


class MeanIoUFromProbabilities(Metric):
    """
    wrapper class around tf.keras.metrics.MeanIoU
    convert softmax probabilities to sparse label
    """
    def __init__(self, num_class):
        super(MeanIoUFromProbabilities, self).__init__()
        self.metric = MeanIoU(num_class)

    # y_true: batch_size * height * width * 1
    # y_predict: batch_size * height * width * num_class
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        self.metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        self.metric.reset_states()


class PrecisionFromProbabilities(Metric):
    def __init__(self):
        super(PrecisionFromProbabilities, self).__init__()
        self.metric = Precision()

    # y_true: batch_size * height * width * 1
    # y_predict: batch_size * height * width * num_class
    def update_state(self, y_true, y_pred, sample_weight=None):
        # make the shape of y_true and y_pred identical
        y_true = tf.reduce_max(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        self.metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        self.metric.reset_states()


class RecallFromProbabilities(Metric):
    def __init__(self):
        super(RecallFromProbabilities, self).__init__()
        self.metric = Recall()

    # y_true: batch_size * height * width * 1
    # y_predict: batch_size * height * width * num_class
    def update_state(self, y_true, y_pred, sample_weight=None):
        # make the shape of y_true and y_pred identical
        y_true = tf.reduce_max(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        self.metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        self.metric.reset_states()

