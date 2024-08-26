from __future__ import absolute_import, division, print_function
import os

import numpy as np
import time
import json
import tqdm
import copy
import cv2
import matplotlib.pyplot as plot
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from depth_tool.ply import write_ply

torch.manual_seed(100)
torch.cuda.manual_seed(100)
from depth_tool.utils.visualize import show_flops_params
from metrics.metrics import compute_depth_metrics, Evaluator

from network.Elite360D import Elite360D_ResNet, Elite360D_effb5, Elite360D_SwinB, Elite360D_SwinT, Elite360D_DilateT
from data_loader.matterport3d import Matterport3D

class Tester:
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        val_dataset = Matterport3D(root_dir='/home/ps/data/haoai/dataset/Matterport3D-1K',
                                   list_file='./splitsm3d/matterport3d_test.txt', disable_color_augmentation=False,
                                   disable_LR_filp_augmentation=False,
                                   disable_yaw_rotation_augmentation=False, is_training=False)

        self.val_loader = DataLoader(val_dataset, batch_size=self.settings.batch_size,
                                     shuffle=False, num_workers=self.settings.num_workers,
                                     pin_memory=True, drop_last=False)

        # network
        if self.settings.model_name == "Elite360D_R18":
            self.model = Elite360D_ResNet(backbone="resnet18", embed_channel=64, ico_nblocks=3, L=1)
        elif self.settings.model_name == "Elite360D_R34":
            self.model = Elite360D_ResNet(backbone="resnet34", embed_channel=64, ico_nblocks=3, L=1)
        elif self.settings.model_name == "Elite360D_R50":
            self.model = Elite360D_ResNet(backbone="resnet50", embed_channel=128, ico_nblocks=3, L=1)
        elif self.settings.model_name == "Elite360D_Effb5":
            self.model = Elite360D_effb5(backbone="effinetb5", embed_channel=64, ico_nblocks=3, L=1)
        elif self.settings.model_name == "Elite360D_SwinB":
            self.model = Elite360D_SwinB(backbone="swinb", embed_channel=128, ico_nblocks=3, L=1)
        elif self.settings.model_name == "Elite360D_SwinT":
            self.model = Elite360D_SwinT(backbone="swint", embed_channel=64, ico_nblocks=3, L=1)
        elif self.settings.model_name == "Elite360D_DilateT":
            self.model = Elite360D_DilateT(backbone="dilateformert", embed_channel=64, ico_nblocks=3, L=1)
        else:
            self.model = None

        self.model.to(self.device)

        try:
            show_flops_params(copy.deepcopy(self.model), self.device)
        except Exception as e:
            print('get flops and params error: {}'.format(e))

        if self.settings.load_weights_dir is None:
            print("No load Model")
            import sys
            sys.exit()

        self.load_model(self.settings.load_weights_dir)

        print("Metrics are saved to:\n", self.log_path)

        self.evaluator = Evaluator(median_align=False)

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb"]:
                inputs[key] = ipt.to(self.device)

        #####
        equi_inputs = inputs["normalized_rgb"]
        ico_images = inputs["ico_img"]
        ico_coords = inputs["ico_coord"]

        outputs = self.model(equi_inputs,ico_images,ico_coords)

        gt = inputs["gt_depth"] * inputs["val_mask"]
        pred = outputs["pred_depth"] * inputs["val_mask"]
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]

        return outputs
    def validate(self):
        """Validate the models on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs = self.process_batch(inputs)
                pred_depth = outputs["pred_depth"].detach() * inputs["val_mask"]
                gt_depth = inputs["gt_depth"] * inputs["val_mask"]
                mask = inputs["val_mask"]
                self.evaluator.compute_eval_metrics(gt_depth, pred_depth,mask)
                if batch_idx % self.settings.log_frequency ==0:
                    self.evaluator.print(dir=self.log_path)
                    rgb_img = inputs["rgb"].detach().cpu().numpy()

                    rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)

                    error = torch.abs(pred_depth - gt_depth) * mask
                    error[error < 0.1] = 0
                    error = error.detach().cpu().numpy()
                    depth_prediction = pred_depth.detach().cpu().numpy()
                    gt_prediction = gt_depth.detach().cpu().numpy()

                    img_dir = os.path.join(self.log_path,"image",str(batch_idx))
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    cv2.imwrite('{}/test_equi_rgb.png'.format(img_dir),
                                rgb_img[:,:,::-1] * 255)
                    plot.imsave('{}/test_equi_depth_pred.png'.format(img_dir),
                                depth_prediction[0, 0, :, :], cmap="jet")
                    plot.imsave('{}/test_equi_gt.png'.format(img_dir),
                                gt_prediction[0, 0, :, :], cmap="jet")
                    plot.imsave('{}/test_error.png'.format(img_dir),
                                error[0, 0, :, :], cmap="jet")
                    # write_ply('{}/test_ico'.format(img_dir), [ico_ori_coords, ico_ori_images[:,::-1]],
                    #           ['x', 'y', 'z', 'blue', 'green', 'red'])
            self.evaluator.print(dir=self.log_path)
        self.evaluator.print(dir=self.log_path)
        del inputs, outputs

    def load_model(self,dir):
        """Load models from disk
        """
        dir = os.path.expanduser(dir)

        assert os.path.isdir(dir), \
            "Cannot find folder {}".format(dir)
        print("loading models from folder {}".format(dir))

        path = os.path.join(dir, "{}.pth".format("best_model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)