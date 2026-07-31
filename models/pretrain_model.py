import torch.nn as nn

from .pointnet2 import PointNet2Encoder
from .projection_head import ProjectionHead


class PretrainRegistrationModel(nn.Module):
    def __init__(self, feat_dim=128, proj_dim=64):
        super().__init__()
        self.src_encoder = PointNet2Encoder(out_channels=feat_dim)
        self.tgt_encoder = PointNet2Encoder(out_channels=feat_dim)
        self.src_projection = ProjectionHead(feat_dim, hidden_dim=feat_dim, out_dim=proj_dim)
        self.tgt_projection = ProjectionHead(feat_dim, hidden_dim=feat_dim, out_dim=proj_dim)

    def encode_src(self, points, batch):
        _, _, _, global_feat = self.src_encoder(points, batch)
        return self.src_projection(global_feat)

    def encode_tgt(self, points, batch):
        _, _, _, global_feat = self.tgt_encoder(points, batch)
        return self.tgt_projection(global_feat)
