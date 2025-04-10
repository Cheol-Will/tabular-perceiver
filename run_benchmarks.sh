#!/bin/bash

# model_type="TabPerceiver"
# default setting: epochs=50 and num_trials=20

# model_types=("TabNet" "ResNet" "TabTransformer" "Trompt" "ExcelFormer")
# model_types=("LightGBM" "TabPerceiver" "TabNet" "FTTransformer" "ResNet" "TabTransformer" "Trompt" "ExcelFormer")
# model_types=("TabNet" "LightGBM" "FTTransformer" "ResNet" "TabTransformer" "Trompt" "ExcelFormer")
model_types=("LightGBM" "FTTransformer" "ResNet" "TabTransformer" "Trompt" "ExcelFormer")


# model_type="TabPerceiver"
task_type="binary_classification"
num_trials=20
epoch=50

scale_types=("small" "medium" "large")
for model_type in "${model_types[@]}"; do
    for scale_type in "${scale_types[@]}"; do
        if [ "$scale_type" == "small" ]; then
            scale_range=$(seq 0 13)
        elif [ "$scale_type" == "medium" ]; then
            scale_range=$(seq 0 8)
        elif [ "$scale_type" == "large" ]; then
            scale_range=$(seq 0 0)
        fi


        for idx in $scale_range; do
            result_path=output/$task_type/$scale_type/$idx/$model_type.pt
            python data_frame_benchmark.py \
            --model_type "$model_type" \
            --task_type "$task_type" \
            --scale "$scale_type" \
            --idx "$idx" \
            --num_trials "$num_trials" \
            --result_path "$result_path" \
            --epoch "$epoch"
        done
    done
done