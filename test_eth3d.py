import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import copy
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

from collections import OrderedDict
from tqdm import tqdm, trange
from PIL import Image
from torch.autograd import Variable
from torch.autograd import grad as Grad
from torchvision import transforms

from models import __models__
from datasets import ETH3D_loader as et
from datasets.data_io import pfm_imread


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DCVSMNet")
parser.add_argument(
    "--model",
    default="DCVSMNet",
    help="select a model structure",
    choices=__models__.keys(),
)
parser.add_argument("--maxdisp", type=int, default=192, help="maximum disparity")
parser.add_argument("--datapath", default="/datasets/ETH3D/", help="data path")
parser.add_argument(
    "--loadckpt",
    default="./checkpoint/sceneflow_gwc_and_norm_correlation.ckpt",
    help="load the weights from a specific checkpoint",
)

parser.add_argument(
    "--cv",
    type=str,
    default="gwc_and_norm_correlation",
    choices=[
        "gwc_and_norm_correlation",
        "gwc_and_concat",
        "gwc_and_gwc_substract",
        "gwc_substract_and_concat",
        "gwc_substract_and_norm_correlation",
        "norm_correlation_and_concat",
    ],
    help="selecting a pair of cost volumes",
)

args = parser.parse_args()

gwc = False
norm_correlation = False
gwc_substract = False
concat = False
if args.cv == "gwc_and_norm_correlation":
    gwc = True
    norm_correlation = True
elif args.cv == "gwc_and_concat":
    gwc = True
    concat = True
elif args.cv == "gwc_and_gwc_substract":
    gwc = True
    gwc_substract = True
elif args.cv == "gwc_substract_and_concat":
    gwc_substract = True
    concat = True
elif args.cv == "gwc_substract_and_norm_correlation":
    gwc_substract = True
    norm_correlation = True
elif args.cv == "norm_correlation_and_concat":
    norm_correlation = True
    concat = True


all_limg, all_rimg, all_disp, all_mask = et.et_loader(args.datapath)


model = __models__[args.model](
    args.maxdisp, gwc, norm_correlation, gwc_substract, concat
)
model = nn.DataParallel(model)
model.cuda()
model.eval()

os.makedirs("./demo/ETH3D/", exist_ok=True)

cv_name = args.loadckpt.split("sceneflow_")[1].split(".")[0]
if cv_name != args.cv:
    raise AssertionError("Please load weights compatible with " + cv_name)

state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict["model"])


pred_mae = 0
pred_op = 0
for i in trange(len(all_limg)):
    limg = Image.open(all_limg[i]).convert("RGB")
    rimg = Image.open(all_rimg[i]).convert("RGB")

    w, h = limg.size
    wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32
    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))

    limg_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(limg)
    rimg_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    disp_gt, _ = pfm_imread(all_disp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
    disp_gt[disp_gt == np.inf] = 0
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    occ_mask = np.ascontiguousarray(Image.open(all_mask[i]))

    with torch.no_grad():

        pred_disp = model(limg_tensor, rimg_tensor, train_status=False)[-1]

        pred_disp = pred_disp[:, hi - h :, wi - w :]

    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 1
    mask = (disp_gt > 0) & (occ_mask == 255)
    error = np.abs(
        predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32)
    )

    pred_error = np.abs(
        predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32)
    )
    pred_op += np.sum(pred_error > op_thresh) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

    filename = os.path.join(
        "./demo/ETH3D/", all_limg[i].split("/")[-2] + all_limg[i].split("/")[-1]
    )
    filename_gt = os.path.join(
        "./demo/ETH3D_gt/", all_limg[i].split("/")[-2] + all_limg[i].split("/")[-1]
    )
    pred_np_save = np.round(predict_np * 4 * 256).astype(np.uint16)
    cv2.imwrite(
        filename,
        cv2.applyColorMap(
            cv2.convertScaleAbs(pred_np_save, alpha=0.01), cv2.COLORMAP_JET
        ),
        [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )
    disp_gt_save = np.round(disp_gt * 4 * 256).astype(np.uint16)
    cv2.imwrite(
        filename_gt,
        cv2.applyColorMap(
            cv2.convertScaleAbs(disp_gt_save, alpha=0.01), cv2.COLORMAP_JET
        ),
        [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )


print(pred_mae / len(all_limg))
print(pred_op / len(all_limg))
