# AlphaHOI

**Seven-in-One**: ECCV'18 (pairwise), CVPR'19 (interactiveness), CVPR'20 (Dj-RN), NeurIPS'20 (IDN), AAAI'21 (DIRV, DecAug), TPAMI(Extended TIN).

AlphaHOI is a project to open the SOTA HOI detection works based on our papers. Currently, it is manintained by [Yong-Lu Li](https://dirtyharrylyl.github.io/) and Xinpeng Liu.

## Papers
- [Extended TIN](https://arxiv.org/abs/2101.10292) (TPAMI'21)
- [DIRV](https://fang-haoshu.github.io/files/DIRV_paper.pdf) (AAAI'21)
- [DecAug](https://fang-haoshu.github.io/files/DecAug_paper.pdf) (AAAI'21)
- [IDN](https://arxiv.org/pdf/2010.16219.pdf) (NeurIPS'20)
- [DJ-RN](https://arxiv.org/pdf/2004.08154.pdf) (CVPR'20)
- [TIN](https://arxiv.org/pdf/1811.08264.pdf) (CVPR'19)
- [Pairwise](https://arxiv.org/pdf/1807.10889) (ECCV'18)

### Results on HICO-DET with different object detections.
|Method| Detector |HAKE| Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[TIN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)| COCO |-| 17.54	|13.80	|18.65|	19.75|	15.70|	20.96|
|TIN| COCO | HAKE-HICO-DET| 22.12 |20.19|22.69|24.06|22.19|24.62|
|TIN| COCO | HAKE-Large| 22.66 |21.17|23.09|24.53|23.00|24.99|
|TIN-PAMI|COCO|-|20.93|18.95|21.32|23.02|20.96|23.42|
|[DJ-RN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/DJ-RN-Torch)| COCO |-| 21.34|18.53|22.18|23.69|20.64|24.60|
|DIRV|COCO+HICO-DET|-|21.78 |16.38| 23.39| 25.52| 20.84| 26.92|
|[IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network))|COCO|-|23.36|22.47|23.63|26.43|25.01|26.85|
|[IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network))|COCO+HICO-DET|-|26.29|22.61|27.39|28.24|24.47|29.37|
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
|DIRV|-|56.1|

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

### TIN (TPAMI'21)
Independent Torch version: [TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network)

### IDN (NeurIPS'20)
Independent Torch version: [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)).

### DIRV (AAAI'21)
Independent Torch version: [DIRV](https://github.com/MVIG-SJTU/DIRV)

### DJ-RN (CVPR'20)
Independent Torch version: [DJ-RN-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/DJ-RN-Torch)

### TIN (CVPR'19)
Independent Torch version: [TIN-Torch](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)


## Citation
If you find our works useful, please consider citing:
```
---TIN-PAMI
@article{li2019transferable,
  title={Transferable Interactiveness Knowledge for Human-Object Interaction Detection},
  author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Huang, Xijie and Xu, Liang and Lu, Cewu},
  journal={TPAMI},
  year={2021}
}
---DIRV
@inproceedings{fang2020dirv,
  title={DIRV: Dense Interaction Region Voting for End-to-End Human-Object Interaction Detection},
  author={Fang, Hao-Shu and Xie, Yichen and Shao, Dian and Lu, Cewu},
  booktitle={AAAI},
  year={2021}
}
---DecAug
@inproceedings{fang2021decaug,
  title={DecAug: Augmenting HOI Detection via Decomposition},
  author={Fang, Hao-Shu and Xie, Yichen and Shao, Dian and Li, Yong-Lu and Lu, Cewu},
  booktitle={AAAI},
  year={2021}
}
---IDN:
@inproceedings{li2020hoi,
  title={HOI Analysis: Integrating and Decomposing Human-Object Interaction},
  author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Li, Yizhuo and Lu, Cewu},
  booktitle={NeurIPS},
  year={2020}
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
---Pairwise
@inproceedings{fang2018pairwise,
  title={Pairwise body-part attention for recognizing human-object interactions},
  author={Fang, Hao-Shu and Cao, Jinkun and Tai, Yu-Wing and Lu, Cewu},
  booktitle={ECCV},
  year={2018}
}
```

## TODO
- [ ] Unifed HOI model
- [ ] TIN-based element analysis
- [ ] Extended DJ-RN

## [HAKE](http://hake-mvig.cn/home/)
**HAKE**[[website]](http://hake-mvig.cn/home/) is a new large-scale knowledge base and engine for human activity understanding. HAKE provides elaborate and abundant **body part state** labels for active human instances in a large scale of images and videos. With HAKE, we boost the action understanding performance on widely-used human activity benchmarks. Now we are still enlarging and enriching it, and looking forward to working with outstanding researchers around the world on its applications and further improvements. If you have any pieces of advice or interests, please feel free to contact [Yong-Lu Li](https://dirtyharrylyl.github.io/) (yonglu_li@sjtu.edu.cn).

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

HAKE-Action-Torch is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
