from __future__ import print_function
import os
import time

import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms
from depth_tool.utils.projection_transformation import get_icosahedron, erp2sphere
from depth_tool.py360convert.E2C import Equirec2Cube


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Matterport3D(data.Dataset):
    """The Matterport3D Dataset"""

    def __init__(self, root_dir, list_file, height=512, width=1024, ico_level=4, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)

        self.w = width
        self.h = height

        self.max_depth_meters = 10.0

        self.ico_level = ico_level

        self.vertices, self.faces = get_icosahedron(self.ico_level)
        self.face_set = self.vertices[self.faces]

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.total_time = 0

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(1024, 512), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)

        gt_depth = gt_depth.astype(np.float32) / 4000

        gt_depth[gt_depth > self.max_depth_meters + 1] = self.max_depth_meters + 1

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        N, _, _ = self.face_set.shape
        t1 = time.time()
        ico_img = erp2sphere(inputs["normalized_rgb"].permute(1, 2, 0).numpy(), np.reshape(self.face_set, [-1, 3]))
        ico_img = np.reshape(ico_img, [N, -1, 3])
        ico_img = np.mean(ico_img, axis=1)
        t2 = time.time()
        self.total_time += t2 - t1

        ico_coord = np.mean(self.face_set, axis=1)

        inputs["ico_img"] = torch.from_numpy(ico_img)
        inputs["ico_coord"] = torch.from_numpy(ico_coord)
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))

        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                              & ~torch.isnan(inputs["gt_depth"]))

        return inputs
