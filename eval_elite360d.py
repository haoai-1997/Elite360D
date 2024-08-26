from __future__ import absolute_import, division, print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse

from Tester.tester_elite360d import Tester

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Training")
# models settings
parser.add_argument("--model_name", type=str, default="",
                    choices=['Elite360D_R18', 'Elite360D_R34', 'Elite360D_R50',
                             'Elite360D_Effb5', 'Elite360D_SwinT', 'Elite360D_SwinB',
                             'Elite360D_DilateT'], help="folder to save the models in")
# system settings
parser.add_argument("--num_workers", type=int, default=32, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")

parser.add_argument("--batch_size", type=int, default=1, help="batch size")

# loading and logging setting
parser.add_argument("--load_weights_dir", default=None, type=str,
                    help="folder of models to load")  # , default='./tmp/panodepth/models/weights_pretrain'
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "tmp"), help="log directory")
parser.add_argument("--log_frequency", type=int, default=200, help="number of batches between each tensorboard log")

args = parser.parse_args()


def main():
    tester = Tester(args)
    tester.validate()


if __name__ == "__main__":
    main()
