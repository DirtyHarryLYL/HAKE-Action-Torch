# HAKE-Activity2Vec(A2V)
General human activity feature extractor and human PaSta (part states) detector based on HAKE data, i.e., HAKE-A2V (Activity2Vec). 
It works like an ImageNet/COCO pre-trained backbone, which aims at extracting the multi-modal activity representation for gerneral downstream tasks like VQA, captioning, cluster, etc. 

PaSta Prediction (93 classes) + Action prediction (156 classes) + Action Vector (Visual & Language) = A2V(image, person box).

The visual feature is based on the human PaSta (Part States from [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf)) recognition, i.e., the features of all human body parts extracted from the PaSta classifiers. 
Meanwhile, the language feature is based on the recognized PaSta scores and their corresponding Bert feature. 
For each PaSta, we will multiply its probability to its Bert vector (base 768) of its PaSta class name. More details can be found in [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf) and [HAKE-Action](https://github.com/DirtyHarryLYL/HAKE-Action).

<p align='center'>
    <img src="demo/a2v-demo.gif", height="400">
</p>

#### Contents in demo
- human ID & box & skeleton
- body part box & states
- human actions

An official demo is coming soon, together with the new version of HAKE-A2V backbone (code, model)!

## Installation
 To install the overall framework of Activity2Vec, please follow [INSTALL.md](./INSTALL.md).

## Dataset
 For the procedure of preparing HAKE dataset for Activity2Vec, please refer to [DATASET.md](./DATASET.md).

## Pretrained Models
 For the download links of the pretrained Activity2Vec models, please refer to [MODEL.md](./MODEL.md).
 
## Getting Started
 To start your journey with Activity2Vec, please refer to [GETTING_STARTED.md](./GETTING_STARTED.md).

## Contributors
 This branch is contributed by Hongwei Fan([@hwfan](https://github.com/hwfan)) and Yong-Lu Li([@DirtyHarryLYL](https://github.com/DirtyHarryLYL)). Please contact them if there are any problems.
 
## Citation
 If you find our works useful, please consider citing:
```
@inproceedings{li2020pastanet,
  title={PaStaNet: Toward Human Activity Knowledge Engine},
  author={Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu and Ma, Ze and Chen, Mingyang and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
```

