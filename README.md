# HAKE_Reasoning_Engine
This repo contains the official implementation of our paper:
**HAKE: A Knowledge Engine Foundation for Human Activity Understanding (TPAMI 2022)**

Yong-Lu Li, Xinpeng Liu, Xiaoqian Wu, Yizhuo Li, Zuoyu Qiu, Liang Xu, Yue Xu, Hao-Shu Fang, Cewu Lu

[[paper](https://arxiv.org/pdf/2202.06851.pdf)]

## Rules
The "PaSta->Activity" rules can be found [here](https://drive.google.com/file/d/1q5EFbyTp-Wb-rAcnRm_Fp0UuRt_paVFC/view?usp=share_link). `read_rules.ipynb` shows an example to read the rules.

## Evaluation
Download the PaSta features [here](https://drive.google.com/file/d/1mSRkD6GDce6E2fLwW6Qey3rLycw2z1aw/view?usp=share_link) and input it into `util/`.

```
export CUDA_VISIBLE_DEVICES=0; \
python train_gt_pasta.py \
--config_path configs/gt-pasta_gt-bbox.yml \
--exp gt-pasta_gt-bbox_eval \
--eval
```
## Citation
```
@article{li2022hake,
  title={Hake: a knowledge engine foundation for human activity understanding},
  author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Li, Yizhuo and Qiu, Zuoyu and Xu, Liang and Xu, Yue and Fang, Hao-Shu and Lu, Cewu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```