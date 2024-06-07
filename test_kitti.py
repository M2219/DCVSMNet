import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as Grad
from torchvision import transforms
import skimage.io
import os
import copy
from collections import OrderedDict
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from datasets import KITTI2015loader as kt2015
from datasets import KITTI2012loader as kt2012
from models import __models__

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='DCVSMNet')
parser.add_argument('--model', default='DCVSMNet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/datasets/kitti_2012/training/", help='data path')
parser.add_argument('--kitti', type=str, default='2012')
parser.add_argument('--loadckpt', default='./checkpoint/sceneflow_gwc_and_norm_correlation.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--cv', type=str, default='gwc_and_norm_correlation', choices=[
          'gwc_and_norm_correlation',
          'gwc_and_concat',
          'gwc_and_gwc_substract',
          'gwc_substract_and_concat',
          'gwc_substract_and_norm_correlation',
          'norm_correlation_and_concat',
], help='selecting a pair of cost volumes')

args = parser.parse_args()

gwc = False
norm_correlation = False
gwc_substract = False
concat = False
if args.cv == 'gwc_and_norm_correlation':
    gwc = True
    norm_correlation = True
elif args.cv == 'gwc_and_concat':
    gwc = True
    concat = True
elif args.cv == 'gwc_and_gwc_substract':
    gwc = True
    gwc_substract = True
elif args.cv == 'gwc_substract_and_concat':
    gwc_substract = True
    concat = True
elif args.cv == 'gwc_substract_and_norm_correlation':
    gwc_substract = True
    norm_correlation = True
elif args.cv == 'norm_correlation_and_concat':
    norm_correlation = True
    concat = True

if args.kitti == '2015':
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2015.kt2015_loader(args.datapath)
else:
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader(args.datapath)

test_limg = all_limg + test_limg
test_rimg = all_rimg + test_rimg
test_ldisp = all_ldisp + test_ldisp

model = __models__[args.model](args.maxdisp, gwc, norm_correlation, gwc_substract, concat)
model = nn.DataParallel(model)
model.cuda()
model.eval()

cv_name = args.loadckpt.split("sceneflow_")[1].split(".")[0]
if cv_name != args.cv:
    raise AssertionError("Please load weights compatible with " + cv_name)

state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

param_n = np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

print("param size = %fMB", param_n)

pred_mae = 0
pred_op = 0
for i in trange(len(test_limg)):
    limg = Image.open(test_limg[i]).convert('RGB')
    rimg = Image.open(test_rimg[i]).convert('RGB')

    w, h = limg.size
    m = 32
    wi, hi = (w // m + 1) * m, (h // m + 1) * m
    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    limg_tensor = transform(limg)
    rimg_tensor = transform(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    disp_gt = Image.open(test_ldisp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32) / 256
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        pred_disp  = model(limg_tensor, rimg_tensor, train_status=False)[-1]
        pred_disp = pred_disp[:, hi - h:, wi - w:]


    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 3
    mask = (disp_gt > 0) & (disp_gt < args.maxdisp)
    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    pred_op += np.sum((pred_error > op_thresh)) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

print("#### EPE", pred_mae / len(test_limg))
print("#### >3.0", pred_op / len(test_limg))
