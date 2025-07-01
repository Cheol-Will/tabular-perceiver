import os
import argparse
import math
import matplotlib.pyplot as plt
from pprint import pprint
from loaders import build_dataset

def main(args):

    dataset = build_dataset(
        task_type=args.task_type,
        dataset_scale=args.scale,
        dataset_index=args.idx
    )
    train_dataset, _, _ = dataset.split()
    col_stats = train_dataset.col_stats
    pprint(col_stats)
    
    df_train = train_dataset.df

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    num_cols = df_train.select_dtypes(include="number").columns.tolist()
    
    num_cols = df_train.select_dtypes(include="number").columns.tolist()
    stats_df = df_train[num_cols].agg(['mean', 'skew']).T  

    avg_skew = stats_df['skew'].abs().mean()
    print("Mean and skewness for each numerical column:")
    print(stats_df)
    print(f"\nAverage skewness across {len(num_cols)} numerical columns: {avg_skew:.4f}")

    
    n = len(num_cols)
    cols_per_row = 3
    rows = math.ceil(n / cols_per_row)

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 4))
    axes = axes.flatten()  

    for ax, col in zip(axes, num_cols):
        ax.hist(df_train[col].dropna(), bins=30)
        ax.set_title(f"{col} Distribution", fontsize=10)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"eda_idx_{args.idx}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved combined histogram figure to '{out_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str,
                        choices=['binary_classification', 'multiclass_classification', 'regression'],
                        default='binary_classification')
    parser.add_argument('--scale', type=str,
                        choices=['small', 'medium', 'large'],
                        default='small')
    parser.add_argument('--idx', type=int,
                        default=0,
                        help='The index of the dataset within DataFrameBenchmark')
    parser.add_argument('--output_dir', type=str,
                        default='output/plots',
                        help='Directory to save histogram PNGs')
    args = parser.parse_args()

    main(args)
