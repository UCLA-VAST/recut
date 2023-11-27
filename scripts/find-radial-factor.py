import pandas as pd
import numpy as np
import argparse

def get_radii(file):
    SWC_COLUMNS = ["id", "type", "x", "y", "z", "radius", "parent_id"]
    swc_df = pd.read_csv(file, sep=r"\s+", comment="#", names=SWC_COLUMNS)
    # print(swc_df.radius)
    return swc_df.radius

def stats(f, g, t):
    scale_factor = f(g) / f(t)
    print(f'Scale factor {f}: {scale_factor}')
    # slope = (1 - scale_factor) / (1 - .4)
    # print(f'Slope: {slope}')
    # return slope

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ground')
    parser.add_argument('test')
    args = parser.parse_args()

    g = get_radii(args.ground)
    t = get_radii(args.test)
    stats(np.mean, g, t)
    stats(np.median, g, t)

if __name__ == "__main__":
    main()
