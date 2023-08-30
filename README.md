# HAKE_Reasoning_Engine
This repo contains the official implementation of our paper:
**HAKE: A Knowledge Engine Foundation for Human Activity Understanding (TPAMI 2022)**

Yong-Lu Li, Xinpeng Liu, Xiaoqian Wu, Yizhuo Li, Zuoyu Qiu, Liang Xu, Yue Xu, Hao-Shu Fang, Cewu Lu

[[paper](https://arxiv.org/pdf/2202.06851.pdf)]

**H**uman **A**ctivity **K**nowledge **E**ngine (**HAKE**) is a novel paradigm to reformulate the human activity understanding task in two-stage: first mapping pixels to an intermediate space spanned by atomic activity primitives, then programming detected primitives with interpretable logic rules to infer semantics.

- **Intermediate primitive space** embeds activity information in images with limited and representative primitives. We build a comprehensive knowledge base by crowdsourcing. As primitive space is sparse, we can cover most primitives from daily activities, i.e., one-time labeling and transferability. The implementation is detailed in master/Activity2Vec branch.
- This branch focuses on the **reasoning engine**, which programs detected primitives into semantics with explicit logic rules and updates the rules during reasoning. That is, diverse activities can be composed of a finite set of primitives via logical reasoning with compositional generalization.

<div align=center><img src="./assert/intro.jpg" width = "80%"></div>

## Logic Rules
As interpretable symbolic reasoning can capture causal primitive-activity relations, we leverage it to program primitives following **logic rules**. A logic rule base is initialized to import common sense: participants are asked to describe the causes (primitives) of effects (activities). Each activity has initial multi-rule from different participants to ensure diversity. 

Some examples are shown below, $P_i, P_j, P_k$ represent *head-eat-sth*, *hand-hold-sth*, *apple* and $A_m$ indicates *eat apple*, a rule is expressed as $P_i\land P_j\land P_k\rightarrow A_m$ ($\land$: AND, $\rightarrow$: implication). $P_i,P_j,P_k,A_m$ are seen as events that are occurring/True or not/False. When $P_i,P_j,P_k$ are True simultaneously, $A_m$ is True. 

<div align=center><img src="./assert/rule.jpg" width = "80%"></div>

The logic rule base can be found [here](https://drive.google.com/file/d/1q5EFbyTp-Wb-rAcnRm_Fp0UuRt_paVFC/view?usp=share_link). `read_rules.ipynb` shows an example to read the rules.


## Reasoning Engine with Logic Rules
For simplicity, we turn $\rightarrow, \land$ into $\vee, \lnot$ via $x\rightarrow y \Leftrightarrow\lnot x\ \vee y$. $\lnot$ and $\vee$ are implemented as functions $NOT(\cdot)$, \ $OR(\cdot,\cdot)$ with Multi-Layer Perceptrons (MLPs) that are reusable for all events. 
We set logic laws (idempotence, complementation, etc.) as optimized objectives imposed on all events to attain logical operations via backpropagation. Then, the expression output is fed into a discriminator to estimate the true probability of an event. 

Given a sample, multi-rule predictions of all activities are generated concurrently. We use voting to combine multi-decision into the final prediction via multi-head attention. 
Besides, to better capture the causal relations, we propose an inductive-deductive policy to make the rule base scalable instead of using rule templates or enumerating possible rules.

<div align=center><img src="./assert/reason.jpg" width = "100%"></div>

## Getting Started
### Evaluation on HICO-DET
Download the PaSta features [here](https://drive.google.com/file/d/1mSRkD6GDce6E2fLwW6Qey3rLycw2z1aw/view?usp=share_link) and input it into `util/`.
Download the checkpoint [here](https://drive.google.com/file/d/1Ta4fpzgjYg_kxo7datMDE70bEdkxAUNO/view?usp=drive_link) and put it into `checkpoint/`.

Then run the following command for evaluation:
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