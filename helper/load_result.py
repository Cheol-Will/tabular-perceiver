from exp_utils import load_result, plot_result, plot_hyperparameter_distribution, load_finetune_result, save_result_dataframe, plot_multitask_history, load_result_multitask
from pprint import pprint
import pandas as pd
import os
import torch

def main():
    ## 

    result_dict = load_result(
        task_type="binary_classification",
        scale_type="small",
    )
    plot_result(
        result_dict=result_dict,
        task_type="binary_classification",
        file_name="deep_vs_lightgbm_250623"
    )
    save_result_dataframe(
    result_dict=result_dict,
    file_name="leaderboard_250623"
    )
    return         

    result_dict = load_result(
        task_type="binary_classification",
    ) 
    print("Result of MemPerceiver")
    for result in result_dict["MemPerceiver"]:
        print(f"{result['idx']}: {result['best_test_metric']} - {result['best_model_cfg']['top_k']}")
    print("\nResult of MemPerceiver_ens")
    for result in result_dict["MemPerceiver_ens"]:
        print(f"{result['idx']}: {result['best_test_metric']} - {result['best_model_cfg']['top_k']}")

    print("\nResult of MemPerceiver_Ens_Attn")
    for result in result_dict["MemPerceiver_Ens_Attn"]:
        print(f"{result['idx']}: {result['best_test_metric']} - {result['best_model_cfg']['top_k']}")


if __name__ == "__main__":
    main()