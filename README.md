# IDN: Integrating-Decomposing Network

#### Code of "HOI Analysis: Integrating and Decomposing Human-Object Interaction" (NeurIPS 2020)
#### Yong-Lu Li*, Xinpeng Liu*, Xiaoqian Wu, Yizhuo Li, Cewu Lu (*=equal contribution).
#### [arXiv]()

As a part of [HAKE-Action-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch) project, you could also use the sub-module of the unified [HAKE-Action-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch) framework.

<p align='center'>
    <img src="misc/demo.png", height="400">
</p>

## Requirements

### Setting up environment

```
pip install -r requirements.txt
```

### Download data

```
bash script/Download_.py
```

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

To get our reported result on HICO-DET, run 

```
python get_map.py
```

## Pre-trained models

For HICO-DET: 

For V-COCO: Comming soon!

## Results

## Contributing

## Acknowledgement

Some of our code are built upon HAKE-Action, analogy and VSGNet.
