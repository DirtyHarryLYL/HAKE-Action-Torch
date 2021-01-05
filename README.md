# HAKE-Activity2Vec (A2V)
General human activity feature extractor and human PaSta (part states) detector based on HAKE data, i.e., HAKE-A2V (Activity2Vec). 
It works like a ImageNet/COCO pre-trained backbone, which aims at extracting the multi-modal activity representation for gerneral downstream tasks like VQA, captioning, cluster, etc. 

PaSta Prediction (93 classes) + Action prediction (156 classes) + Action Vector (Visual & Language) = A2V(image, person box).

The visual feature is based on the human PaSta (Part States from [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf)) recognition, i.e., the features of all human body parts extracted from the PaSta classifiers. 
Meanwhile, the language feature is based on the recognized PaSta scores and their corresponding Bert feature. 
For each PaSta, we will multiply its probability to its Bert vector (base 768) of its PaSta class name. More details can be found in [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf) and [HAKE-Action](https://github.com/DirtyHarryLYL/HAKE-Action).

## Prerequisites
 - Python 3.7
 - CUDA 10.0
 - Anaconda 3(optional)

## Dataset/Models
 For the procedure of preparing HAKE dataset/models for Activity2Vec, please refer to [DATASET.md](./DATASET.md).

## Installation
 ### 1. Create a new conda environment(optional)
 ```
 conda create -y -n activity2vec python=3.7
 conda activate activity2vec
 conda install pip
 ```
 ### 2. Install the dependencies
 ```
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
 git clone https://github.com/DirtyHarryLYL/HAKE-Action-Torch
 cd HAKE-Action-Torch && git checkout Activity2Vec
 pip install -r requirements.txt
 pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
 ```

 ### 3. Setup AlphaPose and Activity2Vec
 ```
 cd AlphaPose && python setup.py build develop && cd ..
 python setup.py build develop
 ```

## Getting Started

Inference with pretrained model on GPU 0 and show the visualization results of the images in demo folder:

```
python -u tools/demo.py --cfg models/a2v/configs/a2v.yaml \
                        --input demo/ \
                        --mode image \
                        --show \
                        GPU_ID 0
```

Load the pretrained ResNet-50 model and finetune the pasta classifier of foot part on GPU 2:

```
python -u tools/train_net.py --cfg models/a2v/configs/foot.yaml \
                             --model finetune-foot \
                             TRAIN.CHECKPOINT_PATH models/a2v/checkpoints/pretrained_res50.pth \
                             MODEL.POSE_MAP True \
                             GPU_ID 2
```

Load the pretrained ResNet-50 model with finetuned pasta classifier and finetune the verb classifier of foot part on GPU 3:

```
python -u tools/train_net.py --cfg models/a2v/configs/verb.yaml \
                             --model finetune-verb \
                             TRAIN.CHECKPOINT_PATH models/a2v/checkpoints/pretrained_model.pth \
                             MODEL.POSE_MAP True \
                             GPU_ID 3
```

Test the finetuned model on the test set with the detection results from Faster-RCNN:

```
python -u tools/test_net.py --cfg models/a2v/configs/verb.yaml \
                            TEST.WEIGHT_PATH models/a2v/configs/pretrained_model.pth \
                            GPU_ID 0
```

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

