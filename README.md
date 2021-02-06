# HAKE-Activity2Vec (A2V)
General human activity feature extractor and human PaSta (part states) detector based on HAKE data, i.e., HAKE-A2V (Activity2Vec). 
It works like a ImageNet/COCO pre-trained backbone, which aims at extracting the multi-modal activity representation for gerneral downstream tasks like VQA, captioning, cluster, etc. 

PaSta Prediction (93 classes) + Action prediction (156 classes) + Action Vector (Visual & Language) = A2V(image, person box).

The visual feature is based on the human PaSta (Part States from [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf)) recognition, i.e., the features of all human body parts extracted from the PaSta classifiers. 
Meanwhile, the language feature is based on the recognized PaSta scores and their corresponding Bert feature. 
For each PaSta, we will multiply its probability to its Bert vector (base 768) of its PaSta class name. More details can be found in [PaStaNet](https://arxiv.org/pdf/2004.00945.pdf) and [HAKE-Action](https://github.com/DirtyHarryLYL/HAKE-Action).

<p align='center'>
    <img src="demo/images/HAKE-A2V.gif", height="400">
</p>

#### Contents in demo
- human ID & box & skeleton
- body part box & states
- human actions

An official demo is coming soon, together with the new version of HAKE-A2V backbone (code, model)!

## Prerequisites
 - Python 3.7
 - CUDA 10.0
 - Anaconda 3(optional)

## Installation
```
 # 1. conda environment(optional)
 conda create -y -n activity2vec python=3.7
 conda activate activity2vec
 conda install pip
 
 # 2. Activity2Vec
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
 git clone https://github.com/DirtyHarryLYL/HAKE-Activity2Vec
 cd HAKE-Activity2Vec
 pip install -r requirements.txt
 pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

 # 3. AlphaPose
 git clone https://github.com/hwfan/AlphaPose
 cd AlphaPose && mkdir detector/yolo/data && python setup.py build develop && cd ..

 # 4. data and weights
 mkdir Data/
 # Download the weights and data presented below:
 ┌────────────────────────────┬────────────────────────────────────────────────────────────────────┐
 | FILENAME                   | URL                                                                |
 ├────────────────────────────┼────────────────────────────────────────────────────────────────────┤
 | yolov3-spp.weights         | https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC |
 | fast_res50_256x192.pth     | https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn |
 | freqs_and_weights.tgz      | https://1drv.ms/u/s!ArUVoRxpBphYgtRf6oOL9VvGxoe-6Q?e=zlCZB4        |
 | hake_40v_test_gt_lmdb.tgz  | https://1drv.ms/u/s!ArUVoRxpBphYgtRgebT9Bu-iuS5uhg?e=gr1ynL        |
 | Test_all_part_lmdb.tgz     | https://1drv.ms/u/s!ArUVoRxpBphYgtRh8L-k6Rk3eaW79w?e=IUphXk        |
 | Trainval_GT_HAKE.tgz       | https://1drv.ms/u/s!ArUVoRxpBphYgtRi42sVpiMPmQon9Q?e=EBOU5a        |
 | Trainval_Neg_HAKE.tgz      | https://1drv.ms/u/s!ArUVoRxpBphYgtRj8acRHK41WSIn8Q?e=oBvDbf        |
 | evaluation_data.tgz        | https://1drv.ms/u/s!ArUVoRxpBphYgtRkzTE3tZVYovtCnA?e=uLxNSA        |
 | pretrained_res50.pth       | https://1drv.ms/u/s!ArUVoRxpBphYgtR2NT5ZOXkccHfw3A?e=EGxXGX        |
 └────────────────────────────┴────────────────────────────────────────────────────────────────────┘
 # and sort them into this data stucture:
 HAKE-Activity2Vec
 ├──AlphaPose
 |  ├──detector/yolo/data/yolov3-spp.weights
 |  └──pretrained_models/fast_res50_256x192.pth
 ├──Data
 |  ├──freqs_and_weights.tgz
 |  ├──hake_40v_test_gt_lmdb.tgz
 |  ├──Test_all_part_lmdb.tgz
 |  ├──Trainval_GT_HAKE.tgz
 |  └──Trainval_Neg_HAKE.tgz
 └──./-Results/evaluation_data.tgz

 # 5. post processing
 cd Data && ls *.tgz | xargs -n1 tar xzvf && rm *.tgz && cd ..
 cd ./-Results && ls *.tgz | xargs -n1 tar xzvf && rm *.tgz && cd ..

 # 6. custom configuration
 # configure lib/ult/data_path.json to your own data path.
```
## Usage

### Activity2Vec Inference
```
python -u tools/main.py --indir YOUR_IMAGE_DIR --pasta-model YOUR_MODEL
```

### Training
```
CUDA_VISIBLE_DEVICES={GPU_ID} python -u tools/train_pasta_net.py --model {MODEL_NAME} --base_lr {LEARNING_RATE} --pasta_trained {PASTA_TRAINED}
```

### Testing
```
CUDA_VISIBLE_DEVICES={GPU_ID} python -u tools/test_pasta_net.py --weight {WEIGHT_PATH}
```

### Finetuning
```
CUDA_VISIBLE_DEVICES={GPU_ID} python -u tools/train_pasta_net.py --model {MODEL_NAME} --train_continue 1 --weight {WEIGHT_PATH} --base_lr {BASE_LR} --pasta_trained {PASTA_TRAINED}
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

