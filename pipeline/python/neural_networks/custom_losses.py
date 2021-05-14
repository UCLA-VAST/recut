import inspect
import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


@tf.function
# generate sample weights from predefined class weights. batch labels are
# shape batch_size * height * width * n_labels.
# label_weights[i] is expected to be weights for label i
def compute_sample_weights_from_class_weights(batch_labels, label_weights):
    assert label_weights is not None
    sample_weights = tf.cast(tf.ones_like(batch_labels), tf.float32)
    sample_weights *= label_weights
    return tf.reduce_prod(sample_weights, axis=-1)


@tf.function
# for each batch dynamically generate sample weights based on
# batch class frequency. batch_labels are expected to be one hot encoded
def compute_sample_weights_from_class_frequencies(batch_labels):
    label_counts = tf.cast(tf.reduce_sum(batch_labels, axis=[0, 1, 2]), tf.float32)
    n = tf.reduce_sum(label_counts)
    epsilon = 1e-6
    label_weights = tf.math.exp(label_counts / n)
    return compute_sample_weights_from_class_weights(batch_labels, label_weights)


# wrapper function to satisfy the tf.keras loss functions signature:
# loss(y_true, y_predict)
def sample_weighted_loss_function(loss_class, n_labels, label_weights=None, static=False):
    assert inspect.isclass(loss_class)
    loss_obj = loss_class(from_logits=False, reduction=tf.losses.Reduction.NONE)
    assert isinstance(loss_obj, tf.losses.Loss)

    # expect batch_label shape as batch_size * height * width * 1
    # squeeze and hot encode
    def wrapped(batch_labels, batch_activation):
        if not batch_labels.dtype == tf.int32:
            batch_labels = tf.cast(batch_labels, tf.int32)
        batch_one_hot = tf.one_hot(tf.squeeze(batch_labels), n_labels)
        if not static:
            sample_weights = compute_sample_weights_from_class_frequencies(batch_one_hot)
        else:
            sample_weights = compute_sample_weights_from_class_weights(batch_one_hot, label_weights)
        loss = loss_obj(batch_labels, batch_activation)
        loss *= sample_weights
        return tf.reduce_mean(loss)
    return wrapped


def dummy_loss(y_true, y_predict):
    return 0.0

