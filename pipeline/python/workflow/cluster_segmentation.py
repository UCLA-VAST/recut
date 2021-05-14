import os
import sys
import time
from gcut.preprocess import Preprocess
from gcut.gcut import GCut


class ClusterSegmentation:
    def __init__(self, swc_path, soma_file_path, scale_z=1, scale_ouput_z=False):
        """
        z_scale parameter passes swc file with scaled z axis to gcut, which
        likely creates better gof values. if scale_ouput_z is true, output
        individual swc files also with scaled z axis. otherwise revert the
        z scaling in output
        """
        self.scale_z = scale_z
        self.scale_output_z = scale_ouput_z
        self.preprocessor = Preprocess(swc_path, soma_file_path, scale_z=self.scale_z)
        self.preprocessor.process()
        self.tree_nodes, self.soma_ids = self.preprocessor.results()
        self.tree_nodes.write_swc_file(swc_path.replace('.swc', '_preprocess.swc'))

        self.knife = GCut()
        print(self.soma_ids)

    def segment(self, conditional='mouse', cell_type='neocortex'):
        self.knife.init(self.tree_nodes, self.soma_ids)
        self.knife.assign(conditional=conditional, cell_type=cell_type)

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        neurons = self.knife.assemble()
        for soma_id, neuron_tree in neurons.items():
            swc_soma_name = os.path.basename(self.preprocessor.swc_path).replace('.swc', '_soma={}.swc'.format(soma_id))
            neuron_tree.aivianize()
            if not self.scale_output_z:
                neuron_tree.scale_coordinates(scale_z=1 / self.scale_z)
            neuron_tree.write_swc_file(os.path.join(out_dir, swc_soma_name))


def main():
    if len(sys.argv) < 3:
        raise ValueError('Usage: python3 cluster_segmentation.py [swc_path] [soma_file_path]')
    swc_path = sys.argv[1]
    soma_file_path = sys.argv[2]
    scale_ouput_z = False

    ##################################
    # no need to change things below #
    ##################################

    start = time.time()
    segmentor = ClusterSegmentation(swc_path, soma_file_path, scale_z=5, scale_ouput_z=scale_ouput_z)
    segmentor.segment()
    out_dir = os.path.join(os.path.dirname(swc_path),
                           os.path.splitext(os.path.basename(swc_path))[0])
    segmentor.save(out_dir)
    end = time.time()
    print('total run time = {} seconds'.format(end - start))


if __name__ == '__main__':
    main()

