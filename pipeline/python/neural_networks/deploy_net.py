import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
from .execution_setting import InferenceSetting
from .net import Net


def deploy_net(setting, deploy_pipeline=False, output_prefix='', mask=None):
    assert isinstance(setting, InferenceSetting)

    # https://github.com/tensorflow/tensorflow/issues/25138
    net = Net(setting, deploy_pipeline=deploy_pipeline)
    net.inference(output_prefix=output_prefix, mask=mask)

""" physical_devices = tf.config.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # using one gpu at the moment
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)"""