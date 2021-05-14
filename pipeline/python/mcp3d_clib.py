import os
import sys
from mcp3d_path import Mcp3dPath

# directory where pymcp3d3.so is located
mcp3d_lib_dir = Mcp3dPath().lib_dir
if not os.path.isdir(mcp3d_lib_dir):
    raise ValueError('directory {} does not exist'.format(mcp3d_lib_dir))
if mcp3d_lib_dir not in sys.path:
    sys.path.insert(0, mcp3d_lib_dir)
try:
    import pymcp3d3 as pymcp3d
except ImportError:
    raise ValueError('can not import pymcp3d module')
