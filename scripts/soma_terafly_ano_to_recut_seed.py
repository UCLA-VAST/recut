from pathlib import Path
from pandas import read_csv
from shutil import rmtree
from math import pi
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace


def main(args: Namespace):
    apo_file = Path(args.apo_file)
    annotations_df = read_csv(apo_file)

    recut = apo_file.parent / 'recut_seeds_from_marker'
    if recut.exists():
        rmtree(recut)
    recut.mkdir(exist_ok=True)

    soma_radius_um = args.default_radius if args.default_radius > 0 else None
    for column in ("x", "y", "z", "volsize"):
        annotations_df[column] = annotations_df[column].round(decimals=0).astype(int)

    for row in annotations_df.itertuples():
        x = row.x * args.voxel_size_x
        y = row.y * args.voxel_size_y
        z = row.z * args.voxel_size_z
        if not soma_radius_um:
            soma_radius_um = row.volsize**(1/3)*3/4/pi
        volume = round(4 / 3 * pi * soma_radius_um ** 3, 3)
        with open(recut / f"marker_{row.x}_{row.y}_{row.z}_{int(volume)}", 'w') as marker_file:
            marker_file.write("# x,y,z,radius_um\n")
            marker_file.write(f"{x},{y},{z},{soma_radius_um}")

    print(f"files saved in {recut.__str__()}")


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert TeraFly apo file to recut seeds (marker files) \n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hong Wei Dong Lab (B.R.A.I.N) \n"
    )
    parser.add_argument("--apo_file", "-a", type=str, required=True,
                        help="Marker (apo) file containing all seed locations.")
    parser.add_argument("--default_radius", "-r", type=float, default=0, required=False,
                        help="Default .")
    parser.add_argument("--voxel_size_x", "-dx", type=float, default=1,
                        help="Image voxel size on x-axis (µm).")
    parser.add_argument("--voxel_size_y", "-dy", type=float, default=1,
                        help="Image voxel size on y-axis (µm).")
    parser.add_argument("--voxel_size_z", "-dz", type=float, default=1,
                        help="Image voxel size on z-axis (µm).")
    main(parser.parse_args())
