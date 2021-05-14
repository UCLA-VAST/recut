import os
import pathlib
import shutil
from abc import ABCMeta


class Mcp3dPathV1:
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


class Mcp3dPathDevV1(Mcp3dPathV1):
    def __init__(self):
        super(Mcp3dPathDevV1, self).__init__(deployed=False)

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


class Mcp3dPathDeployV1(Mcp3dPathV1):
    def __init__(self):
        super(Mcp3dPathDeployV1, self).__init__(deployed=True)

    @property
    def requirements_txt_path(self):
        return os.path.join(self.python_dir, 'requirements.txt')

    @property
    def trained_models_dir(self):
        assert self._deployed, 'only deployed pipelines have known trained_models directory'
        return os.path.join(self.project_dir, 'trained_models')

    @property
    def default_model_name(self):
        for name in os.listdir(self.trained_models_dir):
            if name.startswith('default='):
                return name.replace('default=', '')
        raise ValueError('can not find default model name')

    def trained_model_path(self, model_name=None):
        if not model_name:
            model_name = self.default_model_name
        model_dir = os.path.join(self.trained_models_dir, model_name)
        if not os.path.isdir(model_dir):
            raise ValueError('model {} not found'.format(model_name))
        checkpoint_dir = os.path.join(model_dir, 'model_checkpoint')
        for name in os.listdir(checkpoint_dir):
            if name.endswith('.index'):
                return os.path.join(checkpoint_dir, name.replace('.index', ''))
        raise ValueError(
            'no trained model files found under {}'.format(checkpoint_dir))


def get_mcp3d_path_v1(deployed):
    return Mcp3dPathDeployV1() if deployed else Mcp3dPathDevV1()
