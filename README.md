# TIN: Transferable Interactiveness Network in PyTorch

Under construction.

Code for our **CVPR2019** paper *"Transferable Interactiveness Knowledge for Human-Object Interaction Detection"*.

Link: [[arXiv]](https://arxiv.org/abs/1811.08264)

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{li2019transferable,
  title={Transferable Interactiveness Knowledge for Human-Object Interaction Detection},
  author={Li, Yong-Lu and Zhou, Siyuan and Huang, Xijie and Xu, Liang and Ma, Ze and Fang, Hao-Shu and Wang, Yanfeng and Lu, Cewu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3585--3594},
  year={2019}
}
```

## Introduction
Interactiveness Knowledge indicates whether human and object interact with each other or not. It can be learned across HOI datasets, regardless of HOI category settings. We exploit an Interactiveness Network to learn the general interactiveness knowledge from multiple HOI datasets and perform Non-Interaction Suppression before HOI classification in inference. On account of the generalization of interactiveness, our **TIN: Transferable Interactiveness Network** is a transferable knowledge learner and can be cooperated with any HOI detection models to achieve desirable results. *TIN* outperforms state-of-the-art HOI detection results by a great margin, verifying its efficacy and flexibility.

![Overview of Our Framework](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network/blob/master/images/overview.jpg?raw=true)

## Results on HICO-DET and V-COCO

**Our Results on HICO-DET dataset**

|Method| Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|RC<sub>D</sub>(paper)| 13.75 | 10.23 | 15.45 | 15.34| 10.98|17.02|
|RP<sub>D</sub>C<sub>D</sub>(paper)| 17.03 | 13.42| 18.11| 19.17| 15.51|20.26|
|RC<sub>T</sub>(paper)| 10.61  | 7.78 | 11.45 | 12.47 | 8.87|13.54|
|RP<sub>T1</sub>C<sub>D</sub>(paper)| 16.91   | 13.32 | 17.99 | 19.05 | 15.22|20.19|
|RP<sub>T2</sub>C<sub>D</sub>(paper)| 17.22   | 13.51 | 18.32 | 19.38 | 15.38|20.57|
|Interactiveness-optimized| **17.54**  | **13.80** | **18.65** | **19.75** | **15.70** |**20.96**|

**Our Results on V-COCO dataset**

|Method| Full(def) |
|:---:|:---:|
|RC<sub>D</sub>(paper)| 43.2|
|RP<sub>D</sub>C<sub>D</sub>(paper)| 47.8 |
|RC<sub>T</sub>(paper)| 38.5 |
|RP<sub>T1</sub>C<sub>D</sub>(paper)| 48.3  |
|RP<sub>T2</sub>C<sub>D</sub>(paper)| 48.7 |
|Interactiveness-optimized| **49.0** |

**Please note that we have reimplemented TIN (e.g. replacing the vanilla HOI classifier with iCAN and using cosine_decay lr), thus the result here is different and slight better than the one in [[Arxiv]](https://arxiv.org/abs/1811.08264).**

## Getting Started

### Installation

1.Clone this repository.

```
git clone https://github.com/AndrewZhou924/TIN.torch
```

2.Download dataset and setup evaluation and API. (The detection results (person and object boudning boxes) are collected from: iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection [[website]](http://chengao.vision/iCAN/).)

```
chmod +x ./script/Dataset_download.sh 
./script/Dataset_download.sh
```

3.Install Python dependencies.

```
pip install -r requirements.txt
```

If you have trouble installing requirements, try to update your pip or try to use conda/virtualenv.

### Training

1.Train on HICO-DET dataset

Train from scratch
```
python3 tools/train_HICO.py 
```
Continue training
```
python3 tools/train_HICO.py --train_continue 1 --weight {path_to_your_pretrained_model}
```

### Testing

1.Evaluation on HICO-DET dataset

```
python3 tools/test_HICO.py --weight {path_to_your_pretrained_model}
```

### Change Checkpoint from tensorflow to pytorch

```
python3 tools/changeTfWeightToPytorch.py --ckpt {path_to_your_tf_ckpt} --saveH5 True
```

### Notes on training and Q&A

Since the interactiveness branch is easier to converge, first pre-training the whole model with HOI classification loss only then finetuning with both HOI and interactiveness loss is preferred to get the best performance.

Q: How is the used loss weights generated? 

A: Please refer to this [issue](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network/issues/36) for detailed explanation.

## TODOS
- [ ] CVPR'19 version in PyTorch
- [ ] TPAMI'21 version in PyTorch, PaStaNet-HOI benchmark


## Acknowledgement

Some of the codes are built upon [TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network), [Analogy](https://github.com/jpeyre/analogy) and [VSGNet](https://github.com/ASMIftekhar/VSGNet).
If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

TIN(Transferable Interactiveness Network) is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.


