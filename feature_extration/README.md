# Feature extraction for IDN

## Getting started

### Setting up environment

```shell
pip install -r requirements.txt
```

### Download images

```shell
bash script/Download_set.sh
```

## Extract feature

1. For HICO-DET train set

```shell
export CUDA_VISIBLE_DEVICES=0;python tools/extract.py
export CUDA_VISIBLE_DEVICES=0;python tools/extract.py --output Data/feature/train --mode 1
```

2. For HICO-DET test set, with COCO detector (Optional, you could also straightly download by [this](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)#download-data))

```shell
export CUDA_VISIBLE_DEVICES=0;python tools/extract.py --data Data/db_test_feat.pkl --image_path Data/hico_20160224_det/images/test2015/ --output Data/Union_feature/test/
export CUDA_VISIBLE_DEVICES=0;python tools/extract.py --data Data/db_test_feat.pkl --image_path Data/hico_20160224_det/images/test2015/ --output Data/feature/test/ --mode 1
```

3. Others are comming soon!
