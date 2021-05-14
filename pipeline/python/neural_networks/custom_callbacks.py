import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from .data_io.label_rules import LabelRules
from neural_networks.util.model_utils import ModelUtils


class WritePrediction(Callback):
    # https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/callbacks.py#L450-L456
    # self.model attribute in base class
    def __init__(self, model_util, dataset, label_rules, write_iteration_interval):
        super(WritePrediction, self).__init__()
        assert isinstance(model_util, ModelUtils)
        assert isinstance(dataset, tf.data.Dataset)
        assert isinstance(label_rules, LabelRules)
        self._model_util = model_util
        self._dataset = dataset
        self._label_rules = label_rules
        self._write_iteration_inverval = write_iteration_interval
        self._epoch_id = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_id = epoch

    def on_train_batch_end(self, batch, logs=None):
        if batch % self._write_iteration_inverval > 0:
            return
        for batch_images, batch_labels in self._dataset:
            batch_predictions = self.model.predict_from_inputs(batch_images)
            self._model_util.save_outputs(batch_images, batch_predictions, self._epoch_id, batch,
                                          batch_labels=batch_labels, label_rules=self._label_rules)
            break


class SaveModel(Callback):
    """
    function like tf.keras.callbacks.ModelCheckpoint, only providing options
    to save every save_epoch_inverval epochs instead of saving at each epoch
    """
    def __init__(self, save_path, n_epochs, save_epoch_inverval=1):
        super(SaveModel, self).__init__()
        self._save_path = save_path
        self._n_epochs = n_epochs
        self._save_epoch_interval = save_epoch_inverval

    def on_epoch_end(self, epoch, logs=None):
        """
        save at every self._save_epoch_interval epochs, as well as last epoch
        """
        if epoch + 1 == self._n_epochs or ((epoch + 1) % self._save_epoch_interval == 0):
            if 'val_loss' not in logs:
                logs['val_loss'] = -1.0
            self.model.save_weights(self._save_path.format(epoch=epoch + 1, val_loss=logs['val_loss']))




