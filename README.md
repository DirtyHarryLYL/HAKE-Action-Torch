# HAKE-Action-Torch
HAKE-Action (**PyTorch**) is a project to open the SOTA action understanding studies based on our [Human Activity Knowledge Engine](http://hake-mvig.cn/home/). It includes reproduced SOTA models and their HAKE-enhanced versions.
Currently, it is manintained by [Yong-Lu Li](https://dirtyharrylyl.github.io/), Xinpeng Liu and Zhanke Zhou.

#### **News**: (2020.10.xx) Our xxx is released!

- Paper is [here](https://arxiv.org/abs/2004.00945). 
- We are enriching the data and part states on more data and activities (e.g., upon AVA, more kinds of action categories, more rare actions, etc.). 
- We will keep updating our HAKE-Action model zoo to include more SOTA models and their HAKE-enhanced versions.

## [Data Mode](https://github.com/DirtyHarryLYL/HAKE)
- **HAKE-HICO** (**PaStaNet\* mode** in [paper](https://arxiv.org/abs/2004.00945)): image-level, add the aggression of all part states in an image (belong to one or multiple active persons), compared with original [HICO](http://www-personal.umich.edu/~ywchao/hico/), the only additional labels are image-level human body part states.

- **HAKE-HICO-DET** (**PaStaNet\*** in [paper](https://arxiv.org/abs/2004.00945)): instance-level, add part states for each annotated persons of all images in [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/), the only additional labels are instance-level human body part states.

- **HAKE-Large** (**PaStaNet** in [paper](https://arxiv.org/abs/2004.00945)): contains more than 120K images, action labels and the corresponding part state labels. The images come from the existing action datasets and crowdsourcing. We mannully annotated all the active persons with our novel part-level semantics.

- **GT-HAKE** (**GT-PaStaNet\*** in [paper](https://arxiv.org/abs/2004.00945)): GT-HAKE-HICO and G-HAKE-HICO-DET. It means that we use the part state labels as the part state prediction. That is, we can **perfectly** estimate the body part states of a person. Then we use them to infer the instance activities. This mode can be seen as the **upper bound** of our HAKE-Action. From the results below we can find that, the upper bound is far beyond the SOTA performance. Thus, except for the current study on the conventional instance-level method, continue promoting **part-level** method based on HAKE would be a very promising direction.

- **HAKE-AVA** (**X\*** in [paper]()): Comming soon.

- **HAKE-Object** (**X\*** in [paper])): Comming soon.






## Results on HICO-DET with different object detections.
Faster RCNN, ResNet-101 (1. COCO: pre-trained on COCO, 2. Finetune: finetuned on HICO-DET train set, 3. GT: Ground Truth); enhanced with HAKE.

## Results on V-COCO with different object detections.
Faster RCNN, ResNet-101 (1. COCO: pre-trained on COCO, 2. Finetune: finetuned on HICO-DET train set, 3. GT: Ground Truth); enhanced with HAKE.

## Methods

### [HAKE-Only]() (CVPR 2020)

### [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)) (NeurIPS 2020)
xxx

### [DJ-RN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/DJ-RN-Torch) (CVPR 2020)
xxx

### [TIN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/TIN-Torch) (CVPR 2019)
xxx

## Citation
If you find our work useful, please consider citing:
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
```

## [HAKE](http://hake-mvig.cn/home/)
**HAKE**[[website]](http://hake-mvig.cn/home/) is a new large-scale knowledge base and engine for human activity understanding. HAKE provides elaborate and abundant **body part state** labels for active human instances in a large scale of images and videos. With HAKE, we boost the action understanding performance on widely-used human activity benchmarks. Now we are still enlarging and enriching it, and looking forward to working with outstanding researchers around the world on its applications and further improvements. If you have any pieces of advice or interests, please feel free to contact [Yong-Lu Li](https://dirtyharrylyl.github.io/) (yonglu_li@sjtu.edu.cn).

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

HAKE-Action is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
