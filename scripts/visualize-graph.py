#!/opt/local/bin/python
from pygel3d import hmesh, graph, gl_display as gl
from os import getcwd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', help="*.graph file")
    args = parser.parse_args()

    viewer = gl.Viewer()

    print('Building FEQ...')
    s = graph.load(args.graph)
    viewer.display(s, reset_view=True)
    m_skel = hmesh.skeleton_to_feq(s)#, [5.0]*len(s.nodes()))
    hmesh.cc_split(m_skel)
    hmesh.cc_smooth(m_skel)
    manifold = hmesh.Manifold(m_skel)

    print("Displaying. HIT ESC IN GRAPHICS WINDOW TO PROCEED...")
    viewer.display(m_skel, reset_view=True)

if __name__ == "__main__":
    main()
