from __future__ import absolute_import, division, print_function
import os

import numpy as np
import time
import json
import tqdm
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

torch.manual_seed(100)
torch.cuda.manual_seed(100)
from depth_tool.utils.visualize import show_flops_params
from metrics.metrics import compute_depth_metrics, Evaluator
from losses.losses import BerhuLoss
import losses.loss_gradient as loss_g
from network.Elite360D import Elite360D_ResNet, Elite360D_effb5, Elite360D_SwinB, Elite360D_SwinT, Elite360D_DilateT
from data_loader.matterport3d import Matterport3D


def gradient(x):
    gradient_model = loss_g.Gradient_Net()
    g_x, g_y = gradient_model(x)
    return g_x, g_y


class Trainer_:
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
            torch.cuda.set_device(self.settings.local_rank)

        if len(self.settings.gpu_devices) > 1:
            torch.distributed.init_process_group('nccl', init_method='env://',
                                                 world_size=len(self.settings.gpu_devices),
                                                 rank=self.settings.local_rank)

        self.log_path = os.path.join(self.settings.log_dir,self.settings.model_name)

        if self.settings.local_rank == 0 and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        train_dataset = Matterport3D('/home/ps/data/haoai/dataset/Matterport3D-1K',
                                     './splitsm3d/matterport3d_train.txt',
                                     disable_color_augmentation=self.settings.disable_color_augmentation,
                                     disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                     disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation,
                                     is_training=True)

        self.train_sampler = None if len(
            self.settings.gpu_devices) < 2 else torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=len(self.settings.gpu_devices), rank=self.settings.local_rank)

        self.train_loader = DataLoader(train_dataset, batch_size=self.settings.batch_size,
                                       shuffle=(self.train_sampler is None), num_workers=self.settings.num_workers,
                                       pin_memory=True, sampler=self.train_sampler, drop_last=True)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs

        val_dataset = Matterport3D('/home/ps/data/haoai/dataset/Matterport3D-1K',
                                   './splitsm3d/matterport3d_test.txt',
                                   disable_color_augmentation=self.settings.disable_color_augmentation,
                                   disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                   disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation,
                                   is_training=False)

        self.val_sampler = None if len(
            self.settings.gpu_devices) < 2 else torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=len(self.settings.gpu_devices), rank=self.settings.local_rank)

        self.val_loader = DataLoader(val_dataset, batch_size=self.settings.batch_size,
                                     shuffle=(self.val_sampler is None), num_workers=self.settings.num_workers,
                                     pin_memory=True, sampler=self.val_sampler, drop_last=True)

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

        if len(self.settings.gpu_devices) > 1:
            process_group = torch.distributed.new_group(list(range(len(self.settings.gpu_devices))))
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)
        self.model.to(self.device)

        if len(self.settings.gpu_devices) > 1:
            if self.settings.local_rank == 0:
                try:
                    show_flops_params(copy.deepcopy(self.model), self.device)
                except Exception as e:
                    print('get flops and params error: {}'.format(e))
        else:
            try:
                show_flops_params(copy.deepcopy(self.model), self.device)
            except Exception as e:
                print('get flops and params error: {}'.format(e))

        if len(self.settings.gpu_devices) > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.settings.local_rank],
                                                                   output_device=self.settings.local_rank,
                                                                   find_unused_parameters=False)

        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)

        if self.settings.load_weights_dir is not None:
            self.load_model()
        if self.settings.local_rank == 0:
            print("Training models named:\n ", self.settings.model_name)
            print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
            print("Training is using:\n ", self.device)

        self.compute_loss = BerhuLoss()
        self.evaluator = Evaluator()

        self.writers = {}
        if self.settings.local_rank == 0:
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if self.settings.local_rank == 0:
            self.save_settings()

        self.best_rmse = 20
        self.current_rmse = 0

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.validate()
        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            self.validate()
        self.save_last_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs)

            self.optimizer.zero_grad()

            losses["loss"].backward()

            self.optimizer.step()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.settings.log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:

                mask = inputs["val_mask"]

                pred_depth = outputs["pred_depth"].detach() * mask
                gt_depth = inputs["gt_depth"] * mask

                depth_errors = compute_depth_metrics(gt_depth, pred_depth, mask)
                for i, key in enumerate(self.evaluator.metrics.keys()):
                    losses[key] = np.array(depth_errors[i].cpu())

                if self.settings.local_rank == 0:
                    self.log("train", inputs, outputs, losses)

            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb"]:
                inputs[key] = ipt.to(self.device)

        losses = {}
        # print(inputs["val_mask"].size())

        equi_inputs = inputs["normalized_rgb"]
        ico_images = inputs["ico_img"]
        ico_coords = inputs["ico_coord"]

        outputs = self.model(equi_inputs, ico_images, ico_coords)

        gt = inputs["gt_depth"] * inputs["val_mask"]
        pred = outputs["pred_depth"] * inputs["val_mask"]
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]

        G_x, G_y = gradient(gt.float())
        p_x, p_y = gradient(pred)
        losses["loss"] = (self.compute_loss(inputs["gt_depth"][inputs["val_mask"]],
                                            outputs["pred_depth"][inputs["val_mask"]])
                          + self.compute_loss(G_x, p_x) + self.compute_loss(G_y, p_y))

        return outputs, losses

    def validate(self):
        """Validate the models on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.process_batch(inputs)
                pred_depth = outputs["pred_depth"].detach() * inputs["val_mask"]
                gt_depth = inputs["gt_depth"] * inputs["val_mask"]
                mask = inputs["val_mask"]
                self.evaluator.compute_eval_metrics(gt_depth, pred_depth, mask)
        if self.settings.local_rank == 0:
            self.evaluator.print(dir=self.log_path)

        for i, key in enumerate(self.evaluator.metrics.keys()):
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        if self.settings.local_rank == 0:
            self.log("val", inputs, outputs, losses)
        self.current_rmse = np.array(self.evaluator.metrics["err/rms"].avg.cpu()).item()
        if self.current_rmse < self.best_rmse:
            self.best_rmse = self.current_rmse
            if (self.epoch + 1) % self.settings.save_frequency == 0 and self.settings.local_rank == 0:
                self.save_best_model()
        del inputs, outputs, losses

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
        inputs["gt_depth"] = inputs["gt_depth"] * inputs["val_mask"]
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.settings.batch_size)):  # write a maxmimum of four images
            writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.step)
            writer.add_image("gt_depth/{}".format(j),
                             inputs["gt_depth"][j].data / inputs["gt_depth"][j].data.max(), self.step)
            writer.add_image("pred_depth/{}".format(j),
                             outputs["pred_depth"][j].data / outputs["pred_depth"][j].data.max(), self.step)

    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_best_model(self):
        """Save models weights to disk
        """
        save_folder = os.path.join(self.log_path, "models")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if self.settings.local_rank == 0:
            save_path = os.path.join(save_folder, "{}.pth".format("best_model"))
            to_save = self.model.module.state_dict()
            torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("best_adam"))
            torch.save(self.optimizer.state_dict(), save_path)

    def save_last_model(self):
        """Save models weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if self.settings.local_rank == 0:
            save_path = os.path.join(save_folder, "{}.pth".format("last_model"))
            to_save = self.model.module.state_dict()
            torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("last_adam"))
            torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load models from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        if self.settings.local_rank == 0:
            print("loading models from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("best_model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "best_adam.pth")
        if os.path.isfile(optimizer_load_path):
            if self.settings.local_rank == 0:
                print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            if self.settings.local_rank == 0:
                print("Cannot find Adam weights so Adam is randomly initialized")
