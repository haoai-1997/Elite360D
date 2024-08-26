from __future__ import absolute_import, division, print_function
import os
import argparse

from Trainer.trainer_elite360d import Trainer_

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Training")

# system settings
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")
parser.add_argument('--local_rank', type=int, default=-1, help="the process to be lanched")
# models settings
parser.add_argument("--model_name", type=str, default="",
                    choices=['Elite360D_R18', 'Elite360D_R34', 'Elite360D_R50',
                             'Elite360D_Effb5', 'Elite360D_SwinT', 'Elite360D_SwinB',
                             'Elite360D_DilateT'], help="folder to save the models in")

# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")

# loading and logging settings
parser.add_argument("--load_weights_dir", default=None, type=str,
                    help="folder of models to load")  # , default='./tmp/panodepth/models/weights_pretrain'
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "workdirs"),
                    help="log directory")
parser.add_argument("--log_frequency", type=int, default=400, help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")

# data augmentation settings
parser.add_argument("--disable_color_augmentation", action="store_true", help="if set, do not use color augmentation")
parser.add_argument("--disable_LR_filp_augmentation", action="store_true",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", action="store_true",
                    help="if set, do not use yaw rotation augmentation")

args = parser.parse_args()


def main():
    trainer = Trainer_(args)
    trainer.train()


if __name__ == "__main__":
    main()
