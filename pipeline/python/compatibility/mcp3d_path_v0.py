import os
import pathlib
import shutil


class Mcp3dPathV0:
    def __init__(self, deployed_pipeline=False):
        file_path = os.path.normpath(os.path.realpath(__file__))
        self._file_path = pathlib.Path(file_path)
        assert self._file_path.is_file()
        self._deployed_pipeline = deployed_pipeline

    @property
    def project_dir(self):
        mcp3d_project_path = self._file_path.parents[1]
        assert mcp3d_project_path.is_dir()
        return str(mcp3d_project_path)

    @property
    def pipeline_dir(self):
        return os.path.join(self.project_dir, 'pipelines')

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
    def src_dir(self):
        return os.path.join(self.project_dir, 'src')

    @property
    def local_deploy_build_script_path(self):
        return os.path.join(self.src_dir, 'util', 'local_deploy_build.sh')

    def tagged_pipeline_dir(self, tag):
        d = os.path.join(self.pipeline_dir, 'pipeline_{}'.format(tag))
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        return d

    @property
    def trained_model_dir(self):
        assert self._deployed_pipeline, 'only deployed pipelines have known trained_model directory'
        trained_model_root_dir = os.path.join(self.project_dir, 'trained_model')
        names = os.listdir(trained_model_root_dir)
        if len(names) == 0:
            raise ValueError('no trained model directory found')
        if len(names) > 1:
            raise ValueError('ambiguous trained model directories')
        return os.path.join(trained_model_root_dir, names[0])

    @property
    def trained_model_path(self):
        checkpoint_dir = os.path.join(self.trained_model_dir, 'model_checkpoint')
        for name in os.listdir(checkpoint_dir):
            if name.endswith('.index'):
                return os.path.join(checkpoint_dir, name.replace('.index', ''))
        raise ValueError('no trained model files found under {}'.format(checkpoint_dir))
