from pathlib import Path
from pandas import read_csv

# annotation file that contains soma locations only
annotations = Path(r"Y:\3D_stitched_LS\20210729_SW210318_07_R_HPC_6x_1000z\Ex_642_Em_680_TeraFly_gaussian\somata_stamp_2022_07_18_13_47.ano.apo")
recut = annotations.parent / 'soma_recut'
recut.mkdir(exist_ok=True)
annotations_df = read_csv(annotations)

for column in ("x", "y", "z", "volsize"):
    annotations_df[column] = annotations_df[column].round(decimals=0).astype(int)

for row in annotations_df.itertuples():
    with open(recut/f"marker_{row.x}_{row.y}_{row.z}_{row.volsize}", 'w') as soma_file:
        soma_file.write("# x,y,z\n")
        soma_file.write(f"{row.x},{row.y},{row.z}")
