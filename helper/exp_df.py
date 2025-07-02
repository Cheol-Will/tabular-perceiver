import pandas as pd

def main():
    df = pd.read_csv("output/metric/leaderboard_250623.csv")
    df = df.pivot(index="idx", columns="model_type", values="best_test_metric")
    df = df.round(4)
    df.to_csv("output/metric/leaderboard_250623_pivot.csv", float_format="%.4f")
    print(df)

    return

if __name__ == "__main__":
    main()