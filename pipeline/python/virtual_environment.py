import os
from subprocess import run
from venv import EnvBuilder
from mcp3d_path import Mcp3dPathDeploy


def install_requirements():
    # create bash script to install requirements.txt
    mcp3d_path = Mcp3dPathDeploy()
    venv_sh_path = os.path.join(mcp3d_path.python_dir, 'venv.sh')
    with open(venv_sh_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('source {}/bin/activate\n'.format(mcp3d_path.venv_dir))
        # upgrade pip. on some systems the pip version in the virtual
        # environment is quite low. this cause the tensorflow-2.1.0 install to
        # fail (tensorflow-2.1.0 requires pip > 19.0)
        f.write('python3 -m pip install --upgrade pip\n')
        f.write('echo $(which python3)\n')
        f.write('echo $(python3 -m pip -V)\n')
        f.write('python3 -m pip install -r {}\n'.format(mcp3d_path.requirements_txt_path))
        f.write('deactivate\n')
    # install requirements
    run(['bash', venv_sh_path], check=True)
    # remove bash script
    os.remove(venv_sh_path)


def main():
    # create virtual environment
    env_builder = EnvBuilder(clear=True, with_pip=True)
    env_builder.create(Mcp3dPathDeploy().venv_dir)
    install_requirements()


if __name__ == '__main__':
    main()
