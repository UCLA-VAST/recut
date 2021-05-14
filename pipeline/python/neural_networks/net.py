from pprint import pprint
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
from pipeline_util import timer
from .model_factory import create_model
from .execution_setting import TrainingSetting, InferenceSetting


class Net(object):
    def __init__(self, execution_setting, deploy_pipeline=False):
        super(Net, self).__init__()
        self.training_setting, self.inference_setting = None, None
        self.model_util = None
        # for training phase self.deploy_pipeline is False
        self.deploy_pipeline = False

        assert isinstance(execution_setting, (TrainingSetting, InferenceSetting))
        if isinstance(execution_setting, TrainingSetting):
            self.training_setting = execution_setting
            pprint(self.training_setting.training_setting())
            self.model_util = self.training_setting.get_model_util()
            self.model_util.save_training_setting(self.training_setting.training_setting())
        else:
            self.inference_setting = execution_setting
            pprint(self.inference_setting.inference_setting())
            self.deploy_pipeline = deploy_pipeline
            self.model_util = self.inference_setting.get_model_util(self.deploy_pipeline)
            self.model_util.save_inference_setting(self.inference_setting.inference_setting())

        self.model = None
        if self.training_setting:
            model_name = self.training_setting.model_name
            model_kwargs = self.training_setting.model_kwargs
        else:
            model_name = self.inference_setting.model_name
            model_kwargs = self.inference_setting.model_kwargs
        # model_factory.create_model
        self.model = create_model(model_name, **model_kwargs)

    def train(self):
        assert self.training_setting
        self.model.train(self.training_setting, self.model_util)

    @timer
    def inference(self, output_prefix='', mask=None):
        assert self.inference_setting
        self.model.inference(self.inference_setting, self.model_util, output_prefix=output_prefix, mask=mask)
