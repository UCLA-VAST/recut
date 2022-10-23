import subprocess
import argparse
import re

def parse_range(string):
    if string.contains('-'):
        m = re.match(r'(\d+)(?:-(\d+))?$', string)
        # ^ (or use .split('-'). anyway you like.)
        if not m:
            raise ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
        start = m.group(1)
        end = m.group(2) or start
        return list(range(int(start,10), int(end,10)+1))
    else:
        return list(int(string))


def call_recut(**kwargs):
    exclude = ['image', 'known-seeds'] 
    args = "".join([f"--{k} {v} ".replace('_', '-') for k,v in kwargs.items() if k not in exclude if v])
    cmd = f"recut {kwargs['image']} {args}"
    print(cmd)
    # subprocess.run(cmd, shell=True)
    # return an object that will be a row of a pandas df for soma_recall, soma_precision, neuron yield, neuron precision, run, git commit, voxel size, for this given run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('known-seeds', help='Evaluate recut somas against ground truth')
    parser.add_argument('--min-radius', type=parse_range)
    parser.add_argument('--max-radius', type=parse_range)
    parser.add_argument('--open-denoise', type=parse_range)
    parser.add_argument('--open-steps', type=parse_range)
    parser.add_argument('--close-steps', type=parse_range)
    parser.add_argument('--fg-percent', type=parse_range)
    args = parser.parse_args()

    call_recut(**vars(args))

if __name__ == "__main__":
    main()
