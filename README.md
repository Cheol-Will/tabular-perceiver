# <div align="center">

**A large tabular foundation model that handles new types of data with only minimal modifications.**

</div>

TabPerceiver is a generalization of TabPFN to handle arbitrary inputs and outputs.
The original TabPFN is pre-trained on amounts of synthetic datasets and works on new dataset in a single forward pass.
However, TabPFN cannot handle arbitrary shape of input and output; thus, it uses padding on inputs or slices outputs to match the shape on a spcific task.
 

TabPerceiver is based on Pytorch-Frame and designed to handle arbitrary shape of inputs and outputs. 
