## Preparing HAKE Dataset for Activity2Vec

For the downloading of HAKE dataset, please follow these steps:

1. Prepare a directory with enough free space(more than 60GB) to store the data of HAKE.
```
  mkdir -p /some/data_dir
  ln -s /some/data_dir Data
```

2. Download the image [hake-large.tgz](https://1drv.ms/u/s!ArUVoRxpBphYgtVPpYBkJoJ1x6_HiQ?e=pWdrTY) and annotation [Trainval_HAKE.tgz](https://1drv.ms/u/s!ArUVoRxpBphYgtVN5AQc4LHFXEypDA?e=iNwhuW), [Test_pred_rcnn.tgz](https://1drv.ms/u/s!ArUVoRxpBphYgtVM-Sg05B5CgA7IeA?e=a4674G), [metadata.tar.gz](https://1drv.ms/u/s!ArUVoRxpBphYgtpRmg6ZfuKQ3IWrTA?e=gzFDBB) packages to Data folder, and extract the data from these packages:
```
ls *.* | xargs -n1 tar xzvf
rm *.tgz && rm *.tar.gz && cd ..
```

3. Finally, the structure of the downloaded data should be like this:
```
HAKE-Action-Torch(Activity2Vec)
|_ Data
   |_ hake-large
   |  |_ hico-train
   |  |_ hico-test
   |  |_ ...
   |
   |_ Trainval_HAKE
   |  |_ data.mdb
   |  |_ lock.mdb
   |
   |_ Test_pred_rcnn
   |  |_ data.mdb
   |  |_ lock.mdb
   |
   |_ metadata
      |_ data_path.json
      |_ gt_pasta_data.pkl
      |_ ...
```
