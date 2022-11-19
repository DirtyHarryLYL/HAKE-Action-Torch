# HAKE-Action-Torch

**Eleven-in-One**: CVPR'18 (Part States), CVPR'19 (interactiveness), CVPR'20 (PaStaNet, Dj-RN, SymNet), NeurIPS'20 (IDN), TPAMI(Upgraded TIN, Upgraded SymNet), CVPR'22 (Interactiveness Field), AAAI'22 (mPD), ECCV'22 (PartMap).

<p align='center'>
    <img src="misc/hake_demo.jpg", height="300">
</p>

HAKE-Action-Torch (**PyTorch**) is a project to open the SOTA action understanding studies based on our project: [Human Activity Knowledge Engine](http://hake-mvig.cn/home/). It includes SOTA models and their corresponding HAKE-enhanced versions based on our related papers (CVPR'18/19/20/22, NeurIPS'20, PAMI'21, AAAI'22, ECCV'22). The TensorFlow version of HAKE-Action is [here](https://github.com/DirtyHarryLYL/HAKE-Action).

Currently, it is manintained by [Yong-Lu Li](https://dirtyharrylyl.github.io/), Xinpeng Liu, and Hongwei Fan.

#### **News**: (2022.11.19) We release the interactive object bounding boxes & classes in the interactions within AVA dataset (2.1 & 2.2)! [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA), [[Paper]](https://arxiv.org/abs/2211.07501). BTW, we also release a CLIP-based human body part states recognizer in [CLIP-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/CLIP-Activity2Vec)!

(2022.07.29) Our new work PartMap (ECCV'22) is released! [Paper](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness/blob/main), [Code](https://github.com/DirtyHarryLYL/HAKE-Action-Torch)

(2022.04.23) Two new works on HOI learning are releassed! [Interactiveness Field](https://arxiv.org/abs/2204.07718) (CVPR'22) and a new HOI metric [mPD](https://arxiv.org/abs/2202.09492) (AAAI'22).

(2022.02.14) We release the human body part state labels based on AVA: [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA) and [HAKE 2.0 paper](https://arxiv.org/abs/2202.06851).

(2021.10.06) Our extended version of [SymNet](https://github.com/DirtyHarryLYL/SymNet) is accepted by TPAMI! Paper and code are coming soon.

(2021.2.7) Upgraded [HAKE-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec) is released! Images/Videos --> human box + ID + skeleton + part states + action + representation. [[Description]](https://drive.google.com/file/d/1iZ57hKjus2lKbv1MAB-TLFrChSoWGD5e/view?usp=sharing)
<p align='center'>
    <img src="https://github.com/DirtyHarryLYL/HAKE-Action-Torch/blob/Activity2Vec/demo/a2v-demo.gif", height="400">
</p>

<!-- ## Full demo: [[YouTube]](https://t.co/hXiAYPXEuL?amp=1), [[bilibili]](https://www.bilibili.com/video/BV1s54y1Y76s) -->

(2021.1.15) Our extended version of [TIN (Transferable Interactiveness Network)](https://arxiv.org/abs/2101.10292) is accepted by TPAMI!

(2020.10.27) The code of [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)) ([Paper](https://arxiv.org/abs/2010.16219)) in NeurIPS'20 is released!

## Project
```Branches
HAKE-Action-Torch
  ├──Master Branch                          # Unified pipeline; CVPR'18/20, PaStanet and Part States.
  ├──CLIP-Activity2Vec                      # CLIP-based Part State & Verb Recognizer.
  ├──IDN-(Integrating-Decomposing-Network)  # NeurIPS'20, HOI Analysis: Integrating and Decomposing Human-Object Interaction.
  ├──DJ-RN-Torch                            # CVPR'20, Detailed 2D-3D Joint Representation for Human-Object Interaction.
  ├──TIN-Torch                              # CVPR'19, Transferable Interactiveness Knowledge for Human-Object Interaction Detection.
  └──SymNet-Torch                           # CVPR'20, Symmetry and Group in Attribute-Object Compositions.
```

## Papers
- [DIO](https://arxiv.org/abs/2211.07501)
- [HAKE 2.0](https://arxiv.org/abs/2202.06851)
- [PartMap](https://arxiv.org/abs/2207.14192) (ECCV'22)
- [Interactiveness Field](https://arxiv.org/abs/2204.07718) (CVPR'22)
- [mPD](https://arxiv.org/abs/2202.09492) (AAAI'22)
- [Extended SymNet](https://arxiv.org/abs/2110.04603) (TPAMI'21)
- [Extended TIN](https://arxiv.org/abs/2101.10292) (TPAMI'21)
- [IDN](https://arxiv.org/pdf/2010.16219.pdf) (NeurIPS'20)
- [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf) (CVPR'20)
- [DJ-RN](https://arxiv.org/pdf/2004.08154.pdf) (CVPR'20)
- [SymNet](https://arxiv.org/pdf/2004.00587.pdf) (CVPR'20)
- [TIN](https://arxiv.org/pdf/1811.08264.pdf) (CVPR'19)
- [Part States](http://ai.ucsd.edu/~haosu/papers/cvpr18_partstate.pdf) (CVPR'18)

### Results on HICO-DET with different object detections.
|Method| Detector |HAKE| Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[TIN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)| COCO |-| 17.54	|13.80	|18.65|	19.75|	15.70|	20.96|
|TIN| COCO | HAKE-HICO-DET| 22.12 |20.19|22.69|24.06|22.19|24.62|
|TIN| COCO | HAKE-Large| 22.66 |21.17|23.09|24.53|23.00|24.99|
|TIN-PAMI|COCO|-|20.93|18.95|21.32|23.02|20.96|23.42|
|[DJ-RN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/DJ-RN-Torch)| COCO |-| 21.34|18.53|22.18|23.69|20.64|24.60|
|[IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network))|COCO|-|23.36|22.47|23.63|26.43|25.01|26.85|
|[IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network))|COCO+HICO-DET|-|26.29|22.61|27.39|28.24|24.47|29.37|
|[IF](https://github.com/Foruck/Interactiveness-Field)|COCO+HICO-DET|-| 33.51	|30.30|	34.46|	36.28|	33.16|
|[ParMap](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness)|COCO+HICO-DET|-| 35.15 |33.71| 35.58| 37.56| 35.87| 38.06|
|[TIN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)| GT Pairs |-|34.26|22.90 |37.65|-|-|-|
|[IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network))|GT Pairs|-|43.98|40.27|45.09|-|-|-|

### Results on V-COCO. 
As VCOCO is built on COCO, thus finetuning detector on VCOCO basically contributes marhinally to performance.
|Method | HAKE | AP(role) |
|:---:|:---:|:---:|
|[TIN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)|-|47.8|
|TIN| HAKE-Large | 51.0|
|TIN-PAMI|-|49.1|
|[IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network))|-|53.3|
|[IF](https://github.com/Foruck/Interactiveness-Field)|-|63.0|

### Results on [Ambiguous-HOI](https://github.com/DirtyHarryLYL/DJ-RN).
|Method| mAP |
|:---:|:---:|
|[TIN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)| 8.22 |
|[DJ-RN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/DJ-RN-Torch)| 10.37 |

### Results on [PaStaNet-HOI](https://arxiv.org/abs/2101.10292)
|Method| mAP |
|:---:|:---:|
|TIN-PAMI| 15.38|


## Modules

### PartMap (ECCV'22)
The independent Torch version is in: [PartMap](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness)

### IF (CVPR'22)
The independent Torch version is in: [IF](https://github.com/Foruck/Interactiveness-Field).

### mPD (AAAI'22)
The independent Torch version is in: [mPD](https://github.com/Foruck/OC-Immunity).

### Activity2Vec (CVPR'20)
The independent Torch version is in: [Activity2Vec (A2V)](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec).

### IDN (NeurIPS'20)
The independent Torch version is in: [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)).

### DJ-RN (CVPR'20)
The independent Torch version is in: [DJ-RN-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/DJ-RN-Torch)

### TIN (CVPR'19)
The independent Torch version is in: [TIN-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)

### SymNet (CVPR'20)
Coming soon.

## Citation
If you find our works useful, please consider citing:
```
---HAKE 2.0
@misc{li2022hake,
  title={HAKE: A Knowledge Engine Foundation for Human Activity Understanding}, 
  author={Yong-Lu Li and Xinpeng Liu and Xiaoqian Wu and Yizhuo Li and Zuoyu Qiu and Liang Xu and Yue Xu and Hao-Shu Fang and Cewu Lu},
  year={2022},
  eprint={2202.06851},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
---PartMap
@inproceedings{wu2022mining,
  title={Mining Cross-Person Cues for Body-Part Interactiveness Learning in HOI Detection},
  author={Xiaoqian Wu, Yong-Lu Li, Xinpeng Liu, Junyi Zhang, Yuzhe Wu, Cewu Lu},
  booktitle={ECCV},
  year={2022}
}
---IF
@inproceedings{liu2022interactiveness,
  title={Interactiveness Field in Human-Object Interactions},
  author={Liu, Xinpeng and Li, Yong-Lu and Wu, Xiaoqian and Tai, Yu-Wing and Lu, Cewu and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2022}
}
---mPD
@inproceedings{liu2022highlighting,
  title={Highlighting Object Category Immunity for the Generalization of Human-Object Interaction Detection},
  author={Liu, Xinpeng and Li, Yong-Lu and Lu, Cewu},
  booktitle={AAAI},
  year={2022}
}
---SymNet-PAMI
@article{li2021learning,
  title={Learning Single/Multi-Attribute of Object with Symmetry and Group},
  author={Li, Yong-Lu and Xu, Yue and Xu, Xinyu and Mao, Xiaohan and Lu, Cewu},
  journal={TPAMI},
  year={2021}
}
---TIN-PAMI
@article{li2022transferable,
  title={Transferable Interactiveness Knowledge for Human-Object Interaction Detection},
  author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Huang, Xijie and Xu, Liang and Lu, Cewu},
  journal={TPAMI},
  year={2022}
}
---IDN
@inproceedings{li2020hoi,
  title={HOI Analysis: Integrating and Decomposing Human-Object Interaction},
  author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Li, Yizhuo and Lu, Cewu},
  booktitle={NeurIPS},
  year={2020}
}
---HAKE:
@inproceedings{li2020pastanet,
  title={PaStaNet: Toward Human Activity Knowledge Engine},
  author={Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu and Ma, Ze and Chen, Mingyang and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
@inproceedings{lu2018beyond,
  title={Beyond holistic object recognition: Enriching image understanding with part states},
  author={Lu, Cewu and Su, Hao and Li, Yonglu and Lu, Yongyi and Yi, Li and Tang, Chi-Keung and Guibas, Leonidas J},
  booktitle={CVPR},
  year={2018}
}
---DJ-RN
@inproceedings{li2020detailed,
  title={Detailed 2D-3D Joint Representation for Human-Object Interaction},
  author={Li, Yong-Lu and Liu, Xinpeng and Lu, Han and Wang, Shiyi and Liu, Junqi and Li, Jiefeng and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
---TIN
@inproceedings{li2019transferable,
  title={Transferable Interactiveness Knowledge for Human-Object Interaction Detection},
  author={Li, Yong-Lu and Zhou, Siyuan and Huang, Xijie and Xu, Liang and Ma, Ze and Fang, Hao-Shu and Wang, Yanfeng and Lu, Cewu},
  booktitle={CVPR},
  year={2019}
}
---SymNet
@inproceedings{li2020symmetry,
  title={Symmetry and Group in Attribute-Object Compositions},
  author={Li, Yong-Lu and Xu, Yue and Mao, Xiaohan and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
```

## TODO
<!-- - [ ] TIN-based element analysis -->
- [x] Refined Activity2Vec
<!-- - [ ] Extended DJ-RN
- [ ] SymNet in Torch -->
<!-- - [ ] Unified model (better A2V, early/late fusion, new representation) -->

## [HAKE](http://hake-mvig.cn/home/)
**HAKE**[[website]](http://hake-mvig.cn/home/) is a new large-scale knowledge base and engine for human activity understanding. HAKE provides elaborate and abundant **body part state** labels for active human instances in a large scale of images and videos. With HAKE, we boost the action understanding performance on widely-used human activity benchmarks. Now we are still enlarging and enriching it, and looking forward to working with outstanding researchers around the world on its applications and further improvements. If you have any pieces of advice or interests, please feel free to contact [Yong-Lu Li](https://dirtyharrylyl.github.io/) (yonglu_li@sjtu.edu.cn).

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

HAKE-Action-Torch is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
