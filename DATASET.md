## Preparing HAKE Dataset/Models for Activity2Vec

For the downloading of HAKE dataset/models, please follow these steps:

1. Prepare a directory with enough free space(more than 60GB) to store the data of HAKE.
```
  mkdir -p /some/data_dir
  ln -s /some/data_dir Data
```

2. Download the image [hake-large.tgz](https://1drv.ms/u/s!ArUVoRxpBphYgtVPpYBkJoJ1x6_HiQ?e=pWdrTY) and annotation [Trainval_HAKE.tgz](https://1drv.ms/u/s!ArUVoRxpBphYgtVN5AQc4LHFXEypDA?e=iNwhuW), [Test_pred_rcnn.tgz](https://1drv.ms/u/s!ArUVoRxpBphYgtVM-Sg05B5CgA7IeA?e=a4674G), [metadata.tar.gz](https://1drv.ms/u/s!ArUVoRxpBphYgtYP8dMretXKYIpWRw?e=rlosr5) packages to Data folder, and extract the data:
```
ls *.* | xargs -n1 tar xzvf
rm *.tgz && rm *.tar.gz && cd ..
```

3. Download the pretrained models package [models.tar.gz](https://1drv.ms/u/s!ArUVoRxpBphYgtYRayw7WEVhTf4HHw?e=7tjFTF) to the root folder, and extract the models: 
```
tar xzvf models.tar.gz
rm models.tar.gz
```