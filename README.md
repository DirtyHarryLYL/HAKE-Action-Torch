# IDN: Integrating-Decomposing Network

#### Code of "HOI Analysis: Integrating and Decomposing Human-Object Interaction" (NeurIPS 2020)
#### Yong-Lu Li*, Xinpeng Liu*, Xiaoqian Wu, Yizhuo Li, Cewu Lu (*=equal contribution).
#### [arXiv](https://arxiv.org/abs/2010.16219)

As a part of [HAKE-Action-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch) (work in progress), you could also use the sub-module in unified [HAKE-Action-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch) framework.

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
bash script/Download_dataset.sh
```

### Feature extraction

For ease of reproducing our reported result, we provide the feature of HICO-DET test set with COCO detector in the previous step. We also show how to extract feature from raw images. For more detailed information, please refer to [feature_extraction](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)/feature_extration).

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

To get our reported result on HICO-DET (COCO/VCL/DRG Dectector), run 

```
python eval.py --config_path configs/eval_hico_coco.yml --exp eval_hico_coco
python get_map.py --exp eval_hico_coco
```

```
python eval.py --config_path configs/eval_hico_vcl.yml --exp eval_hico_vcl
python get_map.py --exp eval_hico_vcl
```

```
python eval.py --config_path configs/eval_hico_drg.yml --exp eval_hico_drg
python get_map.py --exp eval_hico_drg
```

V-COCO is coming soon!

## Pre-trained models

For HICO-DET: https://drive.google.com/file/d/19cRqSeRu3Svuc7TA7Zso_uj9F3DwW-zI/view?usp=sharing

For V-COCO: Coming soon!

## Results

Our model achieves the following performance on :

### [HICO-DET](https://paperswithcode.com/sota/human-object-interaction-detection-on-hico) (By GT Dectection, we mean the GT human-object pair box)

|Method|Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
| :-:    |:-:| :-: |:-:|:-:|:-:|:-:|
| IDN (COCO Detector)   | 23.36 | 22.47 | 23.63 | 26.43 | 25.01 | 26.85 |
| IDN (HICO-DET Detector from VCL)   |24.58 | 20.33|25.86|27.89 | 23.64| 29.16|
| IDN (HICO-DET Detector from DRG)   |26.29 | 22.61|27.39|28.24 | 24.47| 29.37|
| IDN (GT Detection)    |43.98 | 40.27 | 45.09 | - | - | - |


### [V-COCO](https://paperswithcode.com/sota/human-object-interaction-detection-on-v-coco)


|Method|AP role, Scenario 1| AP role, Scenario 2 | 
| :-:   |:-: | :-: |
| IDN    |53.3|60.3|

## Tips
1. Sometimes the hinge loss (semi-hard) is tricky and hard to tune (e.g., exploding pointed in [issue-16](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/issues/16#issue-752809776)), as it is sensitive to the samples in a mini-banch. 
Thus, you could also use the Sigmoid cross-entropy loss only. Most of the time, this policy also works good and can achieve performance similar to the well-modulated model with the hinge loss.

2. We will provide the pre-trained models of AE and other modules recently. Thus, users can start from different training phrases for convenience.

3. Sometimes, during training, the GPU utilization maybe very low, this may due to the heavy IO for loading pre-extracted features. So we suggest that you could put these features in SSD to accelate the IO. Then the training would be 10-20 times faster. And the larger batch and multi-gpu training would be possible. Starting from inputting images is also OK, as a result, a batch may be hard to contain multiple humans/obejcts from different images and thus degrade the performance.

## TODOS
- [x] Using different object detections (COCO, COCO+HICO-DET, GT H-O pairs) in testing
- [ ] VCOCO model
- [ ] Intermediate pre-trained models

## Contributing

Feel free to send us email or raise pull requests.

## Acknowledgement

Some of our code are built upon [HAKE-Action](https://github.com/DirtyHarryLYL/HAKE-Action), [Analogy](https://github.com/jpeyre/analogy) and [VSGNet](https://github.com/ASMIftekhar/VSGNet).
