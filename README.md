# Comparing LSTM and its variants for Parts of Speech Tagging problem

Mini project submitted as part of M.Tech course in semester 2.
 
## Setup
This project is based on Python 3 and requires NVIDIA-GPU with CUDA 9.0 to run PyTorch with GPU computing capabilities.

## Dataset
Tree bank dataset was used for the experiments.

## Details
```documents``` folder contains the base research paper(which was referred for the project) and the project report.

LSTM variants used(from the paper): 
- Standard LSTM Cell
- LSTM Cell without the forget gate 
- LSTM Cell without the output gate 
- LSTM Cell without the input gate
- LSTM cell with forget gate bias set to 1

## Code files
- main.py contains the script for the experiments(training and validation) 
- TreeBankDataSet.py contains data pre-processing steps
- pos_tagger.py contains the architecture of generic pos_tagger
- lstm.py contains variants of LSTM Cell mentioned above.

