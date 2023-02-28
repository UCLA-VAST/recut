from pandas import read_csv, concat
from pathlib import Path

source = Path(r"Y:\3D_stitched_LS\20220725_SW220510_02_LS_6x_1000z\Ex_488_Em_525_Terafly_Ano\Somata")
files = [
    r"SW220406_01_LS_6x_1000z_L_Hemisphere_Adriana_stamp_2022_08_30_18_35.ano.apo",
    r"SW220406_01_LS_6x_1000z_R_Hemisphere_Christine_stamp_2022_08_30_17_08.ano.apo",
]
output = "SW220406_01_LS_6x_1000z_combined"
df = concat([read_csv(source/file) for file in files], ignore_index=True).drop_duplicates().reset_index(drop=True)
# for column in df.columns:
#     print(column)
df["##n"] = df.index
apo_file = source/f"{output}.ano.apo"
ano_file = source/f"{output}.ano"
eswc_file = source/f"{output}.ano.eswc"
df.to_csv(apo_file, index=False)
with open(ano_file, "w") as ano:
    ano.write(f"APOFILE={apo_file.name}\n")
    ano.write(f"SWCFILE={apo_file.name[0:-3]}eswc\n")

with open(eswc_file, "w") as eswc:
    eswc.write(
        "#name undefined\n"
        "#comment terafly_annotations\n"
        "#n type x y z radius parent seg_id level mode timestamp TFresindex\n"
    )
