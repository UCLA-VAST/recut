import os
import pathlib
import shutil
from abc import ABCMeta
from trained_models_catalogue import dev_models_catalogue, deploy_models_catalogue, default_options


class Mcp3dPath:
    __metaclass__ = ABCMeta

    def __init__(self, deployed=False):
        file_path = os.path.normpath(os.path.realpath(__file__))
        self._file_path = pathlib.Path(file_path)
        assert self._file_path.is_file()
        self._deployed = deployed

    @property
    def project_dir(self):
        mcp3d_project_path = self._file_path.parents[1]
        assert mcp3d_project_path.is_dir()
        return str(mcp3d_project_path)

    @property
    def bin_dir(self):
        return os.path.join(self.project_dir, 'bin')

    @property
    def lib_dir(self):
        return os.path.join(self.project_dir, 'lib')

    @property
    def python_dir(self):
        return os.path.join(self.project_dir, 'python')

    @property
    def venv_dir(self):
        return os.path.join(self.project_dir, 'mcp3d_venv')

    @property
    def models_catalogue(self):
        return deploy_models_catalogue if self._deployed else dev_models_catalogue


class Mcp3dPathDev(Mcp3dPath):
    def __init__(self):
        super(Mcp3dPathDev, self).__init__(deployed=False)

    @property
    # directory to root of packaged pipelines
    def pipelines_dir(self):
        return os.path.join(self.project_dir, 'pipelines')

    def versioned_pipeline_dir(self, version):
        d = os.path.join(self.pipelines_dir, 'pipeline_{}'.format(version))
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        return d

    @property
    def src_dir(self):
        return os.path.join(self.project_dir, 'src')

    @property
    def local_deploy_build_script_path(self):
        return os.path.join(self.src_dir, 'util', 'local_deploy_build.sh')

    def trained_model_path(self, label_technique=None, model_classes=None):
        if not (label_technique, model_classes) in dev_models_catalogue:
            raise ValueError('no model trained for label technique {} with classes {}'.format(label_technique, model_classes))
        model_path = self.models_catalogue[(label_technique, model_classes)]
        if not os.path.isfile('{}.index'.format(model_path)):
            raise ValueError('no trained model file found at {}'.format(model_path))
        return model_path


class Mcp3dPathDeploy(Mcp3dPath):
    def __init__(self):
        super(Mcp3dPathDeploy, self).__init__(deployed=True)

    @property
    def requirements_txt_path(self):
        return os.path.join(self.python_dir, 'requirements.txt')

    @property
    def trained_models_dir(self):
        assert self._deployed, 'only deployed pipelines have known trained_models directory'
        return os.path.join(self.project_dir, 'trained_models')

    def trained_model_path(self, label_technique=None, model_classes=None):
        if not (label_technique, model_classes) in deploy_models_catalogue:
            print('no model trained for label technique {} with classes {}'.format(label_technique, model_classes))
            label_technique = default_options['label_technique']
            model_classes = default_options['model_classes']
            print('using default model: ({}, {})'.format(label_technique, model_classes))
        model_name = self.models_catalogue[(label_technique, model_classes)]
        model_dir = os.path.join(self.trained_models_dir, model_name)
        if not os.path.isdir(model_dir):
            raise ValueError('model {} not found'.format(model_name))
        checkpoint_dir = os.path.join(model_dir, 'model_checkpoint')
        for name in os.listdir(checkpoint_dir):
            if name.endswith('.index'):
                return os.path.join(checkpoint_dir, name.replace('.index', ''))
        raise ValueError('no trained model files found under {}'.format(checkpoint_dir))


def get_mcp3d_path(deployed):
    return Mcp3dPathDeploy() if deployed else Mcp3dPathDev()
