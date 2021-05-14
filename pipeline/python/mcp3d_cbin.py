import os
from mcp3d_path import Mcp3dPath

# directory where mcp3d c++ executables (e.g. vaa3d_app2) are located
mcp3d_bin_dir = Mcp3dPath().bin_dir
if not os.path.isdir(mcp3d_bin_dir):
    raise ValueError('directory {} does not exist'.format(mcp3d_bin_dir))
app2_bin = os.path.join(mcp3d_bin_dir, 'vaa3d_app2')
if not os.path.isfile(app2_bin):
    raise ValueError('app2 executable {} not found'.format(app2_bin))
