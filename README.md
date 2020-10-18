# HAKE-Action-Torch

Work in progress

HAKE-Action (**PyTorch**) is a project to open the SOTA action understanding studies based on our [Human Activity Knowledge Engine](http://hake-mvig.cn/home/). It includes reproduced SOTA models and their HAKE-enhanced versions based on our six papers in CVPR'18/19/20 and NeurIPS'20.

Currently, it is manintained by [Yong-Lu Li](https://dirtyharrylyl.github.io/), Xinpeng Liu and Zhanke Zhou.

#### **News**: (2020.10.xx) Our xxx is released!

## Project
```Branches
HAKE-Action-Torch
  ├──Master Branch                          # Unified pipeline; CVPR'18/20, PaStanet and Part States.
  ├──IDN-(Integrating-Decomposing-Network)  # NeurIPS'20, HOI Analysis: Integrating and Decomposing Human-Object Interaction.
  ├──DJ-RN-Torch                            # CVPR'20, Detailed 2D-3D Joint Representation for Human-Object Interaction.
  ├──TIN-Torch                              # CVPR'19, Transferable Interactiveness Knowledge for Human-Object Interaction Detection.
  └──SymNet-Torch                           # CVPR'20, Symmetry and Group in Attribute-Object Compositions.
```

## Papers
- [HAKE](https://arxiv.org/pdf/2004.00945.pdf) (CVPR'20)
- [IDN]() (NeurIPS'20)
- [DJ-RN](https://arxiv.org/pdf/2004.08154.pdf) (CVPR'20)
- [SymNet](https://arxiv.org/pdf/2004.00587.pdf) (CVPR'20)
- [TIN](https://arxiv.org/pdf/1811.08264.pdf) (CVPR'19)
- [Part States](http://ai.ucsd.edu/~haosu/papers/cvpr18_partstate.pdf) (CVPR'18)

## Model Zoo
xxx

### Results on HICO-DET with different object detections.
Faster RCNN, ResNet-101 (1. COCO: pre-trained on COCO, 2. Finetune: finetuned on HICO-DET train set, 3. GT: Ground Truth); enhanced with HAKE.

tab

### Results on V-COCO with different object detections.
Faster RCNN, ResNet-101 (1. COCO: pre-trained on COCO, 2. Finetune: finetuned on HICO-DET train set, 3. GT: Ground Truth); enhanced with HAKE.

tab

## Sub-Models

### HAKE-Only (CVPR'20)
Coming soon.

### Activity2Vec (CVPR'20)
Coming soon.

### IDN (NeurIPS'20)
We also provide a **independent** Torch model in [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)) branch, you could use either it or the submodule in our HAKE-Action-Torch.

### DJ-RN (CVPR'20)
[DJ-RN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/DJ-RN-Torch)

### TIN (CVPR'19)
[TIN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)

### SymNet (CVPR'20)
Coming soon.

## Citation
If you find our works useful, please consider citing:
```
@inproceedings{li2020pastanet,
  title={HOI Analysis: Integrating and Decomposing Human-Object Interaction},
  author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Li, Yizhuo and Lu, Cewu},
  booktitle={NeurIPS},
  year={2020}
}
@inproceedings{li2020pastanet,
  title={PaStaNet: Toward Human Activity Knowledge Engine},
  author={Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu and Ma, Ze and Chen, Mingyang and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
@inproceedings{li2020detailed,
  title={Detailed 2D-3D Joint Representation for Human-Object Interaction},
  author={Li, Yong-Lu and Liu, Xinpeng and Lu, Han and Wang, Shiyi and Liu, Junqi and Li, Jiefeng and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
@inproceedings{li2019transferable,
  title={Transferable Interactiveness Knowledge for Human-Object Interaction Detection},
  author={Li, Yong-Lu and Zhou, Siyuan and Huang, Xijie and Xu, Liang and Ma, Ze and Fang, Hao-Shu and Wang, Yanfeng and Lu, Cewu},
  booktitle={CVPR},
  year={2019}
}
@inproceedings{lu2018beyond,
  title={Beyond holistic object recognition: Enriching image understanding with part states},
  author={Lu, Cewu and Su, Hao and Li, Yonglu and Lu, Yongyi and Yi, Li and Tang, Chi-Keung and Guibas, Leonidas J},
  booktitle={CVPR},
  year={2018}
}
@inproceedings{li2020symmetry,
  title={Symmetry and Group in Attribute-Object Compositions},
  author={Li, Yong-Lu and Xu, Yue and Mao, Xiaohan and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
```

## [HAKE](http://hake-mvig.cn/home/)
**HAKE**[[website]](http://hake-mvig.cn/home/) is a new large-scale knowledge base and engine for human activity understanding. HAKE provides elaborate and abundant **body part state** labels for active human instances in a large scale of images and videos. With HAKE, we boost the action understanding performance on widely-used human activity benchmarks. Now we are still enlarging and enriching it, and looking forward to working with outstanding researchers around the world on its applications and further improvements. If you have any pieces of advice or interests, please feel free to contact [Yong-Lu Li](https://dirtyharrylyl.github.io/) (yonglu_li@sjtu.edu.cn).

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

HAKE-Action is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
