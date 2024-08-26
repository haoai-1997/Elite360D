import torch
import torch.nn as nn
from depth_tool.modules.PointNet import PointNetFeaturePropagation, PointNetSetAbstraction, TransformerBlock,TransformerBlock1


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2

class ICOSA_backbone(nn.Module):
    def __init__(self,npoints, nblocks, nneighbor,input_dim,embedding_dim):
        super(ICOSA_backbone,self).__init__()
        self.nblocks = nblocks
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.transformer1 = TransformerBlock1(embedding_dim, embedding_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = embedding_dim * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, embedding_dim, nneighbor))
        self.fc2 = nn.Sequential(
            nn.Linear(embedding_dim * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    def forward(self, x):
        bs,_,_ = x.shape
        xyz = x[..., :3]

        points = self.transformer1(xyz, self.fc1(x))[0]
        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        res = self.fc2(points)
        return res,xyz

class ICO_extractor(nn.Module):
    def __init__(self, channel, level, ico_nblocks, ico_nneighbor):
        super(ICO_extractor, self).__init__()
        self.num_points = (4 ** level) * 20
        self.channel = channel
        self.ico_nblocks = ico_nblocks
        self.ico_nneighbor = ico_nneighbor
        self.input_dim = 6
        self.ICO_backbone = ICOSA_backbone(self.num_points, self.ico_nblocks, self.ico_nneighbor, self.input_dim,
                                           self.channel)

    def forward(self, ico_img, ico_coord):
        assert ico_img.shape[1] == ico_coord.shape[1]
        assert ico_img.shape[1] == self.num_points

        ico_input = torch.cat([ico_coord, ico_img], dim=-1)
        points = self.ICO_backbone(ico_input)
        return points