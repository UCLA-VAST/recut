from pathlib import Path
from pandas import read_csv
import sys

# annotation file that contains soma locations only
annotations = Path(sys.argv[1])
recut = annotations.parent / 'soma_recut'
recut.mkdir(exist_ok=True)
annotations_df = read_csv(annotations)

for column in ("x", "y", "z", "volsize"):
    annotations_df[column] = annotations_df[column].round(decimals=0).astype(int)

for row in annotations_df.itertuples():
    with open(recut/f"marker_{row.x}_{row.y}_{row.z}_{row.volsize}", 'w') as soma_file:
        soma_file.write("# x,y,z\n")
        soma_file.write(f"{row.x},{row.y},{row.z}")
