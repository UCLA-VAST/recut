from pygel3d import graph
from pygel3d import gl_display as gl
from pygel3d import jupyter_display as jd

g=graph.load("/home/kdmarrett/data/dong/run-292/component-0/mesh_graph.gel")

# jd.set_export_mode(True)
# jd.display(g, smooth=False)
viewer = gl.Viewer()
viewer.display(g)
