## Getting Started with Activity2Vec

### Demo
Inference with the pretrained model on GPU 0 and show the visualization results of the images in hico-test set:

```
python -u tools/demo.py --cfg configs/a2v/a2v.yaml \
                        --input Data/hake-large/hico-test \
                        --mode image \
                        --show-res \
                        GPU_ID 0
```

### Train
Load the pretrained ResNet-50 model and finetune the pasta classifier of foot part on GPU 2:

```
python -u tools/train_net.py --cfg configs/a2v/foot.yaml \
                             --model finetune-foot \
                             TRAIN.CHECKPOINT_PATH checkpoints/a2v/pretrained_res50.pth \
                             GPU_ID 2
```

Load the pretrained ResNet-50 model with finetuned pasta classifier and finetune the verb classifier on GPU 3:

```
python -u tools/train_net.py --cfg configs/a2v/verb.yaml \
                             --model finetune-verb \
                             TRAIN.CHECKPOINT_PATH checkpoints/a2v/pretrained_model.pth \
                             GPU_ID 3
```

### Test
Test the finetuned model on the test set with the detection results from Faster-RCNN:

```
python -u tools/test_net.py --cfg configs/a2v/verb.yaml \
                            TEST.WEIGHT_PATH checkpoints/a2v/pretrained_model.pth \
                            GPU_ID 0
```
