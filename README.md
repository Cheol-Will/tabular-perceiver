# <div align="center">

**A large tabular foundation model that handles new types of data with only minimal modifications.**

</div>

TabPerceiver is a generalization of TabPFN to handle arbitrary inputs and outputs.
The original TabPFN is pre-trained on amounts of synthetic datasets and works on new dataset in a single forward pass.
However, TabPFN cannot handle arbitrary shape of input and output; thus, it uses padding on inputs or slices outputs to match the shape on a spcific task.
 

TabPerceiver is based on Pytorch-Frame and designed to handle arbitrary shape of inputs and outputs. 

```
conda env create -f environment.yaml -n torch_frame
```



<div>
  <ul>
    <li><code>data_frame_benchmark.py</code>: Script for benchmarks including GBDT, Deep Learning Models and TabPerceiver</li>
    <li><code>data_frame_benchmark_fewshot.py</code>: Script for running fewshot learning on Lasso and LightGBM</li>
    <li><code>multitask_moe_tune.py</code>: TabPerceiver with mixture of expert and hyperparameter tuning on multitask learning</li>
    <li><code>multitask_moe.py</code>: TabPerceiver with mixture of expert on multitask learning</li>
    <li><code>multtiask_tune.py</code>: TabPerceiver with hyperparameter tuning on multitask learning</li>
    <li><code>multitask.py</code>: TabPerceiver on multitask learning</li>
    <li><code>train_fewshot.py</code>: TabPerceiver on fewshot learning</li>
  </ul>
</div>