import pandas as pd
import exp_utils

alias_dict = {
    "binary_classification": "bc",
    "multiclass_classification": "multi",
    "regression": "reg",
}

def convert_result_dict_to_row(result_dict: dict, model_name: str):
    data = result_dict.get(model_name, [])
    row = {}
    for entry in data:
        t = alias_dict[entry["task_type"]]
        s = entry["scale_type"]
        i = entry["idx"]
        m = entry["best_test_metric"]
        col = f"{t}_{s}_dataset_{i}"
        row[col] = m
    return pd.DataFrame(row, index=[model_name])

def save_rank_csv(model_names, is_save = False):
    # 1) base paper metrics
    df_paper = pd.read_csv("helper/paper_metrics.csv", index_col=0)

    # 2) 여러 모델이면 rows에 모아서 concat
    if isinstance(model_names, list):
        rows = []
        for m in model_names:
            res = exp_utils.load_result(model_type=m)
            df_row = convert_result_dict_to_row(res, m)
            rows.append(df_row)
        df_converted = pd.concat(rows, axis=0)   # ← 여기를 수정
    else:
        res = exp_utils.load_result(model_type=model_names)
        df_converted = convert_result_dict_to_row(res, model_names)

    # 3) 합치기
    df = pd.concat([df_paper, df_converted], axis=0, sort=False)
    df = df.round(3)

    # 4) 모델들이 전부 갖고 있는 컬럼(열)만 필터링
    model_list = model_names if isinstance(model_names, list) else [model_names]
    good_cols = df.loc[model_list].dropna(axis=1).columns  # ← axis=1
    df_filtered = df[good_cols]
    # 5) rank 계산
    rank_df = pd.DataFrame(index=df_filtered.index)
    for col in df_filtered:
        tp = col.split("_")[0]
        ascending = (tp == "reg")   # regression은 낮을수록 좋다고 가정
        rank_df[col] = df_filtered[col].rank(ascending=ascending, method="min")
    
    rank_df["avg_rank"] = rank_df.mean(axis=1)
    rank_df = rank_df.sort_values(by="avg_rank")
    rank_df = rank_df.round(3)
    
    # 6) 저장
    joined = "_".join(model_list) if isinstance(model_list, list) else model_list[0]
    df.to_csv(f"helper/{joined}_metrics.csv")
    rank_df.to_csv(f"helper/{joined}_rank.csv")

    return df, rank_df

def main():
    models = [
        "TabPerceiver",
        "TabPerceiverMultiTask_1",
        "MemPerceiver",
        # "MemGlobalAvgPool",
    ]
    df, rank_df = save_rank_csv(models, is_save=True)
    print(df)
    print(rank_df)

if __name__=="__main__":
    main()
