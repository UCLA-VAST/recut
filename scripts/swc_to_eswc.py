from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from pandas import read_csv


def main(args: Namespace):
    reconstructions_path = Path(args.reconstructions)
    assert reconstructions_path.exists()
    for swc_file in reconstructions_path.glob("*.swc"):
        ano_file = swc_file.parent / (swc_file.name[0:-3] + "ano")
        apo_file = ano_file.parent / (ano_file.name + ".apo")
        eswc_file = ano_file.parent / (ano_file.name + ".eswc")
        if ano_file.exists() or apo_file.exists() or eswc_file.exists():
            print(f"{swc_file.name} is already converted. "
                  f"Please delete ano, apo or eswc files if reconversion is needed.")
            continue

        swc_df = read_csv(swc_file, sep=" ", comment="#", names=("id", "type_id", "x", "y", "z", "radius", "parent_id"),
                          index_col=0)
        for col_name, value in (("seg_id", 0), ("level", 1), ("mode", 0), ("timestamp", 1), ("TFresindex", 1)):
            swc_df[col_name] = value
        print(swc_df.head())

        with ano_file.open('w') as ano:
            ano.write(f"APOFILE={apo_file.name}\n")
            ano.write(f"SWCFILE={eswc_file.name}\n")

        with apo_file.open('w'):
            apo_file.write_text(
                "##n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b")

        with open(eswc_file, 'a'):
            eswc_file.write_text("#")
            swc_df.to_csv(eswc_file, sep=" ", mode="a")


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert swcs exported from Imaris to eswcs that can be read in y-flipped TeraFly images\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hongwei Dong Lab (B.R.A.I.N) \n"
    )
    parser.add_argument("--reconstructions", "-r", type=str, required=True,
                        help="Path folder containing all swc files.")
    main(parser.parse_args())
