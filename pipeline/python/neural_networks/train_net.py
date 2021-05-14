import pprint
import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
from .execution_setting import TrainingSetting
from .util.model_utils import ModelUtils
from .net import Net


def train_net(setting):
    assert isinstance(setting, TrainingSetting)
    # https://github.com/tensorflow/tensorflow/issues/25138
    physical_devices = tf.config.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # using one gpu at the moment
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # this is needed in tf 2.1.0. otherwise in keras.Model.fit multiple threads
    # are used, but the execution states are incorrect. both data iterator and
    # call to model prediction function gives undermined behaviors
    # does not need to be set during inference time
    tf.config.threading.set_inter_op_parallelism_threads(1)

    net = Net(setting)
    net.train()
