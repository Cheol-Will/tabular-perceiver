import pandas as pd
import os

csv_files = [
    "bc_small.csv",
    "bc_medium.csv",
    "bc_large.csv",
    "multi_medium.csv",
    "multi_large.csv",
    "reg_small.csv",
    "reg_medium.csv",
    "reg_large.csv",
]

df_list = []
for fname in csv_files:
    df = pd.read_csv(fname, index_col=0)

    prefix = os.path.splitext(fname)[0]
    df = df.add_prefix(prefix + "_")

    df_list.append(df)

combined_df = pd.concat(df_list, axis=1)
combined_df.to_csv("paper_metrics.csv")

