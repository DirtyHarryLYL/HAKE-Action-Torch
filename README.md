# HAKE-Action-Torch
HAKE-Action in PyTorch

Data: hico-det, vcoco, hake(12w images and ava), ocr(future)

detector:faster rcnn, 101, coco pre-trained, hico train set finetuned, gt (basic three tracks)

backbone: 50, 101, a2v

model:ican, tin, dj-rn, idn, hake(only)

enhanced: x + hake, f(x, hake)=y, boosting performance, especially on rare classes; x = ican, tin, djrn, idn, vcl, more...
