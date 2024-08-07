from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
import gc
import skimage
import skimage.io
import cv2

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets import __datasets__
from models import __models__
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DCVSMNet")
parser.add_argument(
    "--model",
    default="DCVSMNet",
    help="select a model structure",
    choices=__models__.keys(),
)
parser.add_argument("--maxdisp", type=int, default=192, help="maximum disparity")
parser.add_argument(
    "--dataset", default="kitti", help="dataset name", choices=__datasets__.keys()
)
parser.add_argument("--datapath_12", default="/datasets/kitti_2012/", help="data path")
parser.add_argument("--datapath_15", default="/datasets/kitti_2015/", help="data path")
parser.add_argument(
    "--testlist", default="./filenames/kitti12_test.txt", help="testing list"
)
parser.add_argument(
    "--loadckpt",
    default="./checkpoint/kitti_gwc_and_norm_correlation.ckpt",
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

StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath_12, args.datapath_15, args.testlist, False)
TestImgLoader = DataLoader(
    test_dataset, 1, shuffle=False, num_workers=4, drop_last=False
)

model = __models__[args.model](
    args.maxdisp, gwc, norm_correlation, gwc_substract, concat
)
model = nn.DataParallel(model)
model.cuda()

cv_name = args.loadckpt.split("kitti_")[1].split(".")[0]
if cv_name != args.cv:
    raise AssertionError("Please load weights compatible with " + cv_name)

print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict["model"])


save_dir = "./test"


def test():
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.synchronize()
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        torch.cuda.synchronize()
        print(
            "Iter {}/{}, time = {:3f}".format(
                batch_idx, len(TestImgLoader), time.time() - start_time
            )
        )
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(
            disp_est_np, top_pad_np, right_pad_np, left_filenames
        ):
            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)

            fn = os.path.join(save_dir, fn.split("/")[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)

            if False:
                cv2.imwrite(
                    fn,
                    cv2.applyColorMap(
                        cv2.convertScaleAbs(disp_est_uint, alpha=0.01), cv2.COLORMAP_JET
                    ),
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
                )


@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample["left"].cuda(), sample["right"].cuda(), train_status=False)
    return disp_ests[-1]


if __name__ == "__main__":
    test()
