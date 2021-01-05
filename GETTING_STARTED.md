# Getting Started with Activity2Vec

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
