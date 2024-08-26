import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from depth_tool.modules.ICO_encoder import ICO_extractor
from depth_tool.py360convert import equi2pers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Affinity_attention(nn.Module):
    def __init__(self, ico_dim, erp_dim, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.ico_dim = ico_dim
        self.erp_dim = erp_dim
        self.times = 1
        self.scale = qk_scale or erp_dim ** -0.5
        self.q_linear = nn.ModuleList()
        for i in range(self.times):
            q_linear = nn.Linear(erp_dim, erp_dim, bias=False)
            self.q_linear.append(q_linear)

        self.kv_linear = nn.ModuleList()
        for i in range(self.times):
            kv_linear = nn.Linear(ico_dim, erp_dim * 2, bias=False)
            self.kv_linear.append(kv_linear)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(erp_dim, erp_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        for i in range(self.times):
            q_ = self.q_linear[i](q)
            kv_ = self.kv_linear[i](kv)

            B_q, N_q, C_q = q_.shape
            B_kv, N_kv, C_kv = kv_.shape

            kv_ = kv_.reshape(B_kv, -1, 2, C_kv // 2).permute(2, 0, 1, 3)
            k, v = kv_[0], kv_[1]

            attn = (q_ @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            q = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C_q)

        x = self.proj(q)
        x = self.proj_drop(x)

        return x


class Difference_attention(nn.Module):
    def __init__(self, ico_dim, erp_dim, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.ico_dim = ico_dim
        self.erp_dim = erp_dim
        self.times = 1
        self.scale = qk_scale or erp_dim ** -1
        self.q_linear = nn.ModuleList()
        for i in range(self.times):
            q_linear = nn.Linear(erp_dim, erp_dim, bias=False)
            self.q_linear.append(q_linear)

        self.kv_linear = nn.ModuleList()
        for i in range(self.times):
            kv_linear = nn.Linear(ico_dim, erp_dim * 2, bias=False)
            self.kv_linear.append(kv_linear)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(erp_dim, erp_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fc_delta = nn.ModuleList()
        for i in range(self.times):
            fc_delta = nn.Linear(3, erp_dim, bias=False)
            self.fc_delta.append(fc_delta)

    def forward(self, q, q_coord, kv, kv_coord):
        for i in range(self.times):
            q_ = self.q_linear[i](q)
            kv_ = self.kv_linear[i](kv)

            B_kv, N_kv, C_kv = kv_.shape

            kv_ = kv_.reshape(B_kv, -1, 2, C_kv // 2).permute(2, 0, 1, 3)
            k, v = kv_[0], kv_[1]

            pos_enc = self.fc_delta[i](torch.exp(-torch.abs(q_coord[:, :, None] - kv_coord[:, None])))

            attn = torch.exp(-torch.abs(q_[:, :, None] - k[:, None])) + pos_enc

            attn = F.softmax(attn.sum(-1) * self.scale, dim=-1)

            attn = self.attn_drop(attn)

            q = (attn @ v)

        x = self.proj(q)
        x = self.proj_drop(x)

        return x


class EI_Adaptive_Fusion(nn.Module):
    def __init__(self, resolution, ico_level, ico_nblocks, ico_nneighbor, embedding_dim, L=2):
        super(EI_Adaptive_Fusion, self).__init__()
        self.resolution = resolution
        self.embed_channel = embedding_dim
        self.ico_level = ico_level
        self.ico_nblocks = ico_nblocks
        self.ico_nneighbor = ico_nneighbor
        self.L = L
        self.ICO_encoder = ICO_extractor(self.embed_channel, self.ico_level, self.ico_nblocks, self.ico_nneighbor)

        self.A_attn_bottle = Affinity_attention(ico_dim=self.embed_channel, erp_dim=self.embed_channel)

        self.D_attn_bottle = Difference_attention(ico_dim=self.embed_channel, erp_dim=self.embed_channel)

        self.gate_A_bottle = nn.Linear(self.embed_channel * 2, self.embed_channel, bias=False)
        self.gate_D_bottle = nn.Linear(self.embed_channel * 2, self.embed_channel, bias=False)

        self.sigmoid = nn.Sigmoid()

    def calculate_erp_coord(self, erp_feature):
        def coords2uv(coords, w, h, fov=None):
            # output uv size w*h*2
            uv = torch.zeros_like(coords, dtype=torch.float32)
            middleX = w / 2 + 0.5
            middleY = h / 2 + 0.5
            if fov == None:
                uv[..., 0] = (coords[..., 0] - middleX) / w * 2 * np.pi
                uv[..., 1] = (coords[..., 1] - middleY) / h * np.pi
            else:
                fov_h, fov_w = pair(fov)
                uv[..., 0] = (coords[..., 0] - middleX) / w * (fov_w / 360) * 2 * np.pi
                uv[..., 1] = (coords[..., 1] - middleY) / h * (fov_h / 180) * np.pi
            return uv

        def uv2xyz(uv):
            sin_u = torch.sin(uv[..., 0])
            cos_u = torch.cos(uv[..., 0])
            sin_v = torch.sin(uv[..., 1])
            cos_v = torch.cos(uv[..., 1])
            return torch.stack([
                cos_v * sin_u,
                sin_v,
                cos_v * cos_u,
            ], dim=-1)

        if len(erp_feature.shape) == 4:
            bs, channel, h, w = erp_feature.shape
        else:
            bs, hxw, channel = erp_feature.shape
            h = int(np.sqrt(hxw // 2))
            w = h * 2
        erp_yy, erp_xx = torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w), indexing='ij')
        screen_points = torch.stack([erp_xx, erp_yy], -1)
        erp_coordinate = uv2xyz(coords2uv(screen_points, w, h))
        erp_coordinate = erp_coordinate[None, ...].repeat(bs, 1, 1, 1).to(erp_feature.device)

        erp_coordinate = rearrange(erp_coordinate, "n h w c-> n (h w) c")
        return erp_coordinate

    def forward(self, erp_feature, erp_downsample_list, ico_rgb, ico_coord):
        erp_coordinate = self.calculate_erp_coord(erp_feature)

        bs, c, h, w = erp_feature.shape
        erp_feature = rearrange(erp_feature, 'n c h w -> n (h w) c')

        ## ICO_process
        _, N_ico, Len = ico_rgb.shape
        ICOSA_embedding_set, ICOSA_coordinate = self.ICO_encoder(ico_rgb, ico_coord)

        for i in range(self.L):
            A_fusion_feature = self.A_attn_bottle(erp_feature, ICOSA_embedding_set)

            D_fusion_feature = self.D_attn_bottle(erp_feature, erp_coordinate, ICOSA_embedding_set, ICOSA_coordinate)

            A_factor = self.sigmoid(self.gate_A_bottle(torch.cat([A_fusion_feature, D_fusion_feature], dim=-1)))
            #
            D_factor = self.sigmoid(self.gate_D_bottle(torch.cat([A_fusion_feature, D_fusion_feature], dim=-1)))

            fused_feature = A_factor * A_fusion_feature + D_factor * D_fusion_feature

        fused_feature = rearrange(fused_feature, "n (h w) c-> n c h w", h=h)

        return fused_feature, erp_downsample_list
