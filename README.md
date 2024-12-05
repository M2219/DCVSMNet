<p align="center">
  <h1 align="center">DCVSMNet: Double Cost Volume Stereo Matching Network</h1>
  <p align="center">
    Mahmoud Tahmasebi* (mahmoud.tahmasebi@research.atu.ie), Saif Huq, Kevin Meehan, Marion McAfee
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/2402.16473.pdf">Paper</a>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/M2219/DCVSMNet/blob/main/imgs/DCVSMNet.png" alt="Logo" width="90%">
  </a>
</p>


# Performance on KITTI raw dataset: RTX 4070 S
<p align="center">
  <img width="600" height="300" src="./imgs/mygif.gif" data-zoomable>
</p>

Performance on Jetson AGX Orin for low resolution input
<p align="center">
  <img width="400" height="400" src="./imgs/myimage2.gif" data-zoomable>
</p>

# SOTA results.
The results on SceneFlow

<p align="center"><img width=90% src="imgs/performance.png"></p>


The results on KITTI dataset using RTX 3090.
| Method | KITTI 2012 <br> (3-noc) | KITTI 2012 <br> (3-all) | KITTI 2015 <br> (D1-bg) | KITTI 2015 <br> (D1-fg) | KITTI 2015 <br> (D1-all) |Runtime <br> (ms)|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CGI-Stereo | 1.41 % | 1.76 % | 1.66 % | 3.38 % | 1.94 % | 29 |
| CoEx | 1.55 % | 1.93 % | 1.79 % | 3.82 % | 2.13 % | 33 |
| BGNet+ | 1.62 % | 2.03 % | 1.81 % | 4.09 % | 2.19 % | 35 |
| Fast-ACVNet+ | 1.45 % | 1.85 % | 1.70 % | 3.53 % | 2.01 % | 45 |
| HITNet | 1.41 % | 1.89 % | 1.74 % | **3.20 %** | 1.98 % | 54 |
| DispNetC | 4.11 % | 4.65 % | 2.21 % | 6.16 % | 4.43 % | 60 |
| AANet | 1.91 % | 2.42 % | 1.99 % | 5.39 % | 2.55 % | 62 |
| JDCNet | 1.64 % | 2.11 % | 1.91 % | 4.47 % | 2.33 % | 80 |
| **DCVSMNet**| **1.30 %** | **1.67 %** | **1.60 %** | 3.33 % | **1.89 %** | 67 |

The results on SceneFlow dataset based on the selected cost volumes.
| Group-wise <br> correlation | Norm <br> correlation  | Concatenation | Group-wise <br> substraction |EPE[px] | D1-all[%] |Runtime <br> (ms)|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| &check; | &check; |         |         | 0.60 | 2.11 | 67 |
| &check; |         | &check; |         | 0.59 | 2.05 | 75 |
| &check; |         |         | &check; | 0.59 | 2.06 | 89 |
|         | &check; | &check; |         | 0.72 | 2.59 | 60 |
|         | &check; |         | &check; | 0.65 | 2.28 | 74 |
|         |         | &check; | &check; | 0.69 | 2.38 | 81 |

# How to use

## Environment
* NVIDIA RTX 3090
* Python 3.11
* Pytorch 2.0.0

## Install

### Create a virtual environment and activate it.

```
conda create -n DCVSMNet python=3.11
conda activate DCVSMNet
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4
```

## Data Preparation
* [SceneFlow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)

## Train

Use the following command to train DCVSMNet on SceneFlow.
First training,
```
python train_sceneflow.py --logdir ./checkpoints/sceneflow/first/
```
Second training,
```
python train_sceneflow.py --logdir ./checkpoints/sceneflow/second/ --loadckpt ./checkpoints/sceneflow/first/checkpoint_000059.ckpt
```

Use the following command to finetune DCVSMNet on KITTI using the pretrained model on SceneFlow,
```
python train_kitti.py --logdir ./checkpoints/kitti/ --loadckpt ./checkpoints/sceneflow/second/checkpoint_000059.ckpt
```


### Pretrained Model
* [DCVSMNet](https://drive.google.com/drive/folders/1VcfEpO9Mv0Bt7Xdvckii4SAavW8cL1_d)

Generate disparity images of KITTI test set,
```
python save_disp.py
```

# Citation

If you find this project helpful in your research, welcome to cite the paper.

```
@article{TAHMASEBI2025129002,
title = {DCVSMNet: Double Cost Volume Stereo Matching Network},
journal = {Neurocomputing},
volume = {618},
pages = {129002},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.129002},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224017739},
author = {Mahmoud Tahmasebi and Saif Huq and Kevin Meehan and Marion McAfee},
keywords = {Stereo matching, Cost volume construction, Disparity estimation},
abstract = {We introduce the Double Cost Volume Stereo Matching Network (DCVSMNet11The source code is available at https://github.com/M2219/DCVSMNet.), a novel architecture characterized by two upper (group-wise correlation) and lower (norm correlation) small cost volumes. Each cost volume is processed separately, and a coupling module is proposed to fuse the geometry information extracted from the upper and lower cost volumes. DCVSMNet is a fast stereo matching network with a 67 ms inference time and strong generalization ability which can produce competitive results compared to state-of-the-art methods. The results on several benchmark datasets show that DCVSMNet achieves better accuracy than methods such as CGI-Stereo and BGNet at the cost of greater inference time.}
}
```

# Acknowledgements

Thanks to open source works: [CoEx](https://github.com/antabangun/coex), [ACVNet](https://github.com/gangweiX/Fast-ACVNet), [CGI-Stereo](https://github.com/gangweiX/CGI-Stereo).
