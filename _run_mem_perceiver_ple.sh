task_type="binary_classification"
scale="small"
idx_list=(0)
epochs=1
num_trials=1
num_repeats=1

# idx_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)
# epochs=50
# num_trials=20
# num_repeats=1
exp_name="MemPer_PLE"

for idx in "${idx_list[@]}"; do
    mkdir -p "output/$task_type/$scale/$idx"    
    python train_mem_perceiver_ple.py\
        --task_type "$task_type"\
        --scale "$scale"\
        --idx "$idx"\
        --epochs "$epochs"\
        --num_trials "$num_trials"\
        --num_repeats "$num_repeats"\
        --result_path "output/$task_type/$scale/$idx/$exp_name.pt"\
        --exp_name "$exp_name"| tee "output/$task_type/$scale/$idx/$exp_name.log"
done;