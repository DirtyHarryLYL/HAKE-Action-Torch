## Pretrained Models of Activity2Vec

### Results

We evaluate the performance of the pretrained model on **PaSta and verb detection** tasks, and report the mAP results on each task. We also report the results when "no_interaction" is not included to show the detection performance only when specific interaction occurs. 

More explorations of more advanced models, better utilization of data, transfer learning, leveraging HAKE-A2v on other tasks, etc. are welcome! Please feel free to contact us or pull a request.

|  Task         | w/ no_interaction(mAP) | w/o no_interaction(mAP) |
|    ----       | ---- | ---- |
|  PaSta: foot  | 16.99 | 14.74 |
|  PaSta: leg   | 14.34 | 11.35 |
|  PaSta: hip   | 34.27 | 32.12 |
|  PaSta: hand  | 7.82  | 6.94  |
|  PaSta: arm   | 33.85 | 29.38 |
|  PaSta: head  | 18.92 | 16.36 |
|  **PaSta: avg**   | **21.03** | **18.48** |
|  **verb**         | **12.23** | **12.26** |

### Download Instruction
You could download the pretrained models of Activity2Vec in the following two ways:

1. All-in-one package

    Download the [checkpoints.tar.gz](https://1drv.ms/u/s!ArUVoRxpBphYgtdrKhKcYB2tUaSCIg?e=5c69YL) to the root directory of HAKE-Action-Torch and extract the weights.
    ```
    tar xzvf checkpoints.tar.gz
    rm checkpoints.tar.gz
    ```

2. Single files

    Download the files listed in the table,

    |  File Name  | Description |
    |    ----     |     ----      |
    |  [pretrained_res50.pth](https://1drv.ms/u/s!ArUVoRxpBphYgtdpNrCZmkAWc2e09A?e=u7gPTu)  | The backbone weights pretrained by HAKE-Large data. It can be used to finetune the PaSta classifier from random initialization. |
    |  [pretrained_model.pth](https://1drv.ms/u/s!ArUVoRxpBphYgtdsloRx5CNBosUW-w?e=Dzq7rv)  | The whole weights pretrained by HAKE-Large data. It can be used to infer the human activities and extract the corresponding Activity2Vec features from a raw image. |
    |  [yolov3-spp.weights](https://1drv.ms/u/s!ArUVoRxpBphYgtdt_fcADRpQtT_F2Q?e=JHNEG2)    | Pretrained YOLO weights for AlphaPose inference. More details can be found in https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md |
    |  [fast_res50_256x192.pth](https://1drv.ms/u/s!ArUVoRxpBphYgtdq7kP6LV_lzF3H9w?e=vGc9C9)  | Pretrained pose estimator weights for AlphaPose inference. More details can be found in https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md |
    |  [osnet.pth](https://1drv.ms/u/s!ArUVoRxpBphYgtdo_Gjz3yQI3EiU7w?e=4rLswB)  | Pretrained weights for person re-id in AlphaPose inference. More details can be found in https://github.com/MVIG-SJTU/AlphaPose/tree/master/trackers |

    and sort them to the structure beside:

    ```
    HAKE-Action-Torch(Activity2Vec)
    |_ checkpoints
       |_ a2v
       |  |_ pretrained_model.pth
       |  |_ pretrained_res50.pth
       |
       |_ pose
       |  |_ fast_res50_256x192.pth
       |
       |_ yolo
          |_ yolov3-spp.weights
          |_ osnet.pth
    ```
