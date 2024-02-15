<p align="center">
  <h1 align="center">DCVSMNet: Double Cost Volume Stereo Matching Network</h1>
  <p align="center">
    Mahmoud Tahmasebi*, Saif Huq, Kevin Meehan, Marion McAfee
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/addpath.pdf">Paper</a>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/M2219/DCVSMNet/blob/main/imgs/DCVSMNet.png" alt="Logo" width="90%">
  </a>
</p>


# SOTA results.
The results on SceneFlow

<p align="center"><img width=90% src="imgs/performance.png"></p>


The results on KITTI dataset using RTX 3090.
| Method | KITTI 2012 <br> (3-noc) | KITTI 2012 <br> (3-all) | KITTI 2015 <br> (D1-bg) | KITTI 2015 <br> (D1-fg) | KITTI 2015 <br> (D1-all) |Runtime <br> (ms) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| CGI-Stereo | 1.41 % | 1.76 % | 1.66 % | 3.38 % | 1.94 % | 29|
| CoEx | 1.55 % | 1.93 % | 1.79 % | 3.82 % | 2.13 % |33|
| BGNet+ | 1.62 % | 2.03 % | 1.81 % | 4.09 % | 2.19 % |35|
| Fast-ACVNet+ | 1.45 % | 1.85 % | 1.70 % | 3.53 % | 2.01 % |45|
| HITNet | 1.03 % | 1.34 % | 1.31 % | 3.08 % | 1.61 % |54|
| DispNetC | 4.11 % | 4.65 % | 2.21 % | 6.16 % | 4.43 % |60|
| AANet | 1.91 % | 2.42 % | 1.99 % | 5.39 % | 2.55 % |62|
| JDCNet | 1.64 % | 2.11 % | 1.91 % | 4.47 % | 2.33 % |80|
| DCVSMNet| 1.03 % | 1.34 % | 1.31 % | 3.08 % | 1.61 % |67|


# How to use

## Environment
* NVIDIA RTX 3090
* Python 3.8
* Pytorch 1.12

## Install

### Create a virtual environment and activate it.

```
conda create -n CGI python=3.8
conda activate CGI
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
* [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)

## Train

Use the following command to train CGI-Stereo on Scene Flow.
First training,
```
python train_sceneflow.py --logdir ./checkpoints/sceneflow/first/
```
Second training,
```
python train_sceneflow.py --logdir ./checkpoints/sceneflow/second/ --loadckpt ./checkpoints/sceneflow/first/checkpoint_000019.ckpt
```

Use the following command to train CGI-Stereo on KITTI (using pretrained model on Scene Flow),
```
python train_kitti.py --logdir ./checkpoints/kitti/ --loadckpt ./checkpoints/sceneflow/second/checkpoint_000019.ckpt
```




## Evaluation on Scene Flow and KITTI

### Pretrained Model
* [CGI-Stereo](https://drive.google.com/drive/folders/15pVddbGU6ByYWRWB_CFW2pzANU0mzdU5?usp=share_link)
* [CGF-ACV](https://drive.google.com/drive/folders/1sSZctBVYQzCpG_OPFTPIDonDRkWwca3t?usp=share_link)

Generate disparity images of KITTI test set,
```
python save_disp.py
```

# Citation

If you find this project helpful in your research, welcome to cite the paper.

```
@article{xu2023cgi,
  title={CGI-Stereo: Accurate and Real-Time Stereo Matching via Context and Geometry Interaction},
  author={Xu, Gangwei and Zhou, Huan and Yang, Xin},
  journal={arXiv preprint arXiv:2301.02789},
  year={2023}
}
```

# Acknowledgements

Thanks to Antyanta Bangunharcana for opening source of his excellent work [Correlate-and-Excite](https://github.com/antabangun/coex).
