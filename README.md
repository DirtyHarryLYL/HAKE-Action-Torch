# Code of "HOI Analysis: Integrating and Decomposing Human-Object Interaction" (NeurIPS 2020)

## IDN: Integrating-Decomposing Network

## Prerequisites

Set up environment by ·pip install -r requirements.txt·

## Getting started

### 1. AE pre-train

```shell
export CUDA_VISIBLE_DEVICES=0;python train.py --exp AE --config_path configs/AE.yml
```

### 2. IDN training without IPT (Inter-pair transformation)

```shell
export CUDA_VISIBLE_DEVICES=0;python train.py --exp IDN --config_path configs/IDN.yml
```

### 3. IDN finetuning with IPT

```shell
export CUDA_VISIBLE_DEVICES=0;python train.py --exp IPT --config_path configs/IPT.yml
```


### 4. Evaluation

The evaluation is performed during training. 
Due to the maximum file size limitation of the supplementary material, we only provide toy data to enable the code to run successfully. 
This means the above commands would not produce our reported results. 
To get our reported result on HICO-DET, run 

```
python get_map.py
```
