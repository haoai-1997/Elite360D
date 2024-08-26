import torch
import torch.nn as nn
import copy

from depth_tool.modules.Swin.Swin_transformer import SwinTransformer
from depth_tool.modules.Swin.Swin_transformer_v2 import SwinTransformerV2

class SwinB(nn.Module):
    def __init__(self, pretrained=False):
        super(SwinB, self).__init__()
        # compute healpixel
        self.vit_encoder = SwinTransformer(pretrain_img_size=224,
                                           embed_dim=128,
                                           depths=[2, 2, 18, 2],
                                           num_heads=[4, 8, 16, 32],
                                           window_size=7,
                                           drop_path_rate= 0.5,
                                           frozen_stages=-1)
        if pretrained:
            self.init_weights("checkpoints/swin_base_patch4_window7_224_22k.pth")

    def init_weights(self, pretrained_model):
        self.vit_encoder.init_weights(pretrained_model)

    def forward(self, x):
        out = self.vit_encoder(x)
        return out

class SwinT(nn.Module):
    def __init__(self, pretrained=True, img_size=256, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=16):
        super().__init__()
        self.swin_unet = SwinTransformerV2(img_size=img_size, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                           window_size=window_size)
        if pretrained == True:
            self.load_from()

    def forward(self, rgb):
        output = self.swin_unet(rgb)
        return output

    def load_from(self, ):
        pretrained_path = "checkpoints/swinv2_tiny_patch4_window16_256.pth"

        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model" not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            # print(msg)
            return
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")

        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3 - int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k: v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape,
                                                                              model_dict[k].shape))
                    del full_dict[k]
            else:
                print("{} not in current model.".format(k))
        msg = self.swin_unet.load_state_dict(full_dict, strict=False)