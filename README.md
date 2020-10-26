# HAKE-Action-Torch

Under construction.

<p align='center'>
    <img src="misc/hake_demo.jpg", height="300">
</p>

HAKE-Action-Torch (**PyTorch**) is a project to open the SOTA action understanding studies based on our project: [Human Activity Knowledge Engine](http://hake-mvig.cn/home/). It includes SOTA models and their corresponding HAKE-enhanced versions based on our six papers (CVPR'18/19/20, NeurIPS'20).

Currently, it is manintained by [Yong-Lu Li](https://dirtyharrylyl.github.io/), Xinpeng Liu and Zhanke Zhou, Hongwei Fan.

#### **News**: (2020.10.27) The code of [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)) in NeurIPS'20 is released!

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
- [IDN]() (NeurIPS'20)
- [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf) (CVPR'20)
- [DJ-RN](https://arxiv.org/pdf/2004.08154.pdf) (CVPR'20)
- [SymNet](https://arxiv.org/pdf/2004.00587.pdf) (CVPR'20)
- [TIN](https://arxiv.org/pdf/1811.08264.pdf) (CVPR'19)
- [Part States](http://ai.ucsd.edu/~haosu/papers/cvpr18_partstate.pdf) (CVPR'18)

## Model Zoo
Coming soon.

### Results on HICO-DET with different object detections.
Coming soon.


### Results on V-COCO with different object detections.
Coming soon.


## Modules
All these modules wii be integrated into the master branch. Now the master branch covers HAKE, TIN.

### HAKE-Only (CVPR'20)
Coming soon.

### Activity2Vec (CVPR'20)
The independent Torch version is in: [Activity2Vec (A2V)](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec).

### IDN (NeurIPS'20)
The independent Torch version is in: [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)).

### DJ-RN (CVPR'20)
The independent Torch version is in: [DJ-RN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/DJ-RN-Torch)

### TIN (CVPR'19)
The independent Torch version is in: [TIN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch)

### SymNet (CVPR'20)
Coming soon.

## Citation
If you find our works useful, please consider citing:
```
---IDN:
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

## [HAKE](http://hake-mvig.cn/home/)
**HAKE**[[website]](http://hake-mvig.cn/home/) is a new large-scale knowledge base and engine for human activity understanding. HAKE provides elaborate and abundant **body part state** labels for active human instances in a large scale of images and videos. With HAKE, we boost the action understanding performance on widely-used human activity benchmarks. Now we are still enlarging and enriching it, and looking forward to working with outstanding researchers around the world on its applications and further improvements. If you have any pieces of advice or interests, please feel free to contact [Yong-Lu Li](https://dirtyharrylyl.github.io/) (yonglu_li@sjtu.edu.cn).

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

HAKE-Action is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
