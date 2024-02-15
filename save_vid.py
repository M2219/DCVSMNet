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
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import cv2
from PIL import Image
from datasets.data_io import read_all_lines

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='DCVSMNet')
parser.add_argument('--model', default='DCVSMNet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath_raw', default="/datasets/kittiraw2/", help='data path')
parser.add_argument('--testlist',default='./filenames/kitti_raw.txt', help='testing list')
parser.add_argument('--loadckpt', default='./checkpoint/kitti.ckpt',help='load the weights from a specific checkpoint')
# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath_raw, args.datapath_raw, args.testlist, training=False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

###load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


save_dir = './test'


def load_path(list_filename):
    lines = read_all_lines(list_filename)
    splits = [line.split() for line in lines]
    left_images = [x[0] for x in splits]
    right_images = [x[1] for x in splits]

    return left_images, right_images



def test():
    os.makedirs(save_dir, exist_ok=True)
    fps_list = np.array([])
    for batch_idx, sample in enumerate(TestImgLoader):


        left_filenames, right_filenames = load_path(args.testlist)

        left_name = left_filenames[batch_idx].split('/')[1]
        left_img = np.array(Image.open(os.path.join(args.datapath_raw, left_filenames[batch_idx]))) # .convert('RGB')

        disp_gen, fps = test_sample(sample)
        disp_est_np = tensor2numpy(disp_gen)

        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader), fps))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est, dtype=np.float32)

            fn = os.path.join(save_dir, fn.split('/')[-1])

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            disp_np = cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET)

            print("saving to", fn, disp_est.shape)
            out_img = np.concatenate((left_img, disp_np), 0)
            cv2.putText(out_img, "%.1f fps" % (fps), (10, left_img.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(fn, out_img)

# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()


    im_left = sample['left'].cuda()
    im_right = sample['right'].cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    disp_ests = model(im_left, im_right)
    end.record()
    torch.cuda.synchronize()
    runtime = start.elapsed_time(end)
    fps = 1000/runtime
    return disp_ests[-1], fps


if __name__ == '__main__':
    test()
