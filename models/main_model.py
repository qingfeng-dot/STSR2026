# # models/main_model.py (modified version)

# import torch
# import torch.nn as nn
# from torch_geometric.utils import to_dense_batch
# from .pointnet2 import PointNet2Encoder
# from .registration_head import SVDHead

# class RegistrationModel(nn.Module):
#     def __init__(self, feat_dim=128):
#         super().__init__()
#         self.feat_dim = feat_dim

#         self.src_encoder = PointNet2Encoder(out_channels=feat_dim)
#         self.tgt_encoder = PointNet2Encoder(out_channels=feat_dim)

#         self.head = SVDHead(in_dim_src=feat_dim, in_dim_tgt=feat_dim)

#     def forward(self, p_src, p_tgt):
#         # --- Input and format conversion (no change) ---
#         B, N_s, _ = p_src.shape
#         _, N_t, _ = p_tgt.shape

#         p_src_flat = p_src.view(-1, 3)
#         batch_src_orig = torch.arange(B, device=p_src.device).repeat_interleave(N_s)

#         p_tgt_flat = p_tgt.view(-1, 3)
#         batch_tgt_orig = torch.arange(B, device=p_tgt.device).repeat_interleave(N_t)

#         # --- Encoder (now receives correct return values) ---
#         f_src_sampled, p_src_sampled, batch_src_sampled, _ = self.src_encoder(p_src_flat, batch_src_orig)
#         f_tgt_sampled, p_tgt_sampled, batch_tgt_sampled, _ = self.tgt_encoder(p_tgt_flat, batch_tgt_orig)

#         # --- Reshape using correct batch indices ---
#         # f_src_sampled: [B * N_s_sampled, C]
#         # batch_src_sampled: [B * N_s_sampled]
#         # Now their dimensions match!
#         f_src, _ = to_dense_batch(f_src_sampled, batch_src_sampled, fill_value=0.0, max_num_nodes=512)
#         p_src_ds, _ = to_dense_batch(p_src_sampled, batch_src_sampled, fill_value=0.0, max_num_nodes=512)

#         # Do the same for target
#         # Assume target has 512 points after downsampling; adjust max_num_nodes if needed
#         f_tgt, _ = to_dense_batch(f_tgt_sampled, batch_tgt_sampled, fill_value=0.0, max_num_nodes=512)
#         p_tgt_ds, _ = to_dense_batch(p_tgt_sampled, batch_tgt_sampled, fill_value=0.0, max_num_nodes=512)

#         # --- Dimension check ---
#         # f_src: [B, 512, C]
#         # p_src_ds: [B, 512, 3]
#         # f_tgt: [B, 512, C]
#         # p_tgt_ds: [B, 512, 3]
#         # Dimensions are now correct and dense tensors are ready for use

#         # --- SVD Head (input remains unchanged) ---
#         transform_pred = self.head(p_src_ds, f_src, p_tgt_ds, f_tgt)

#         return transform_pred

# models/main_model.py (modified version)

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from .pointnet2 import PointNet2Encoder
from .registration_head import (
    CrossAttentionSVDHead,
    HybridDirectRegistrationHead,
    SVDHead,
)

class RegistrationModel(nn.Module):
    def __init__(
        self,
        feat_dim=128,
        target_dense_points=512,
        robust_matching=False,
        match_temperature=0.1,
        registration_head="svd",
        attention_heads=4,
        attention_dropout=0.1,
        direct_translation_range=2.0,
        target_saliency_prior_weight=1.0,
        translation_residual_range=0.0,
        translation_residual_detach_features=True,
        translation_residual_hidden_dim=32,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.target_dense_points = int(target_dense_points)
        self.registration_head_type = str(registration_head)

        self.src_encoder = PointNet2Encoder(out_channels=feat_dim)
        self.tgt_encoder = PointNet2Encoder(out_channels=feat_dim)

        if self.registration_head_type == "hybrid_direct":
            self.head = HybridDirectRegistrationHead(
                feature_dim=feat_dim,
                num_heads=int(attention_heads),
                dropout=float(attention_dropout),
                match_temperature=match_temperature,
                target_saliency_prior_weight=float(
                    target_saliency_prior_weight
                ),
                translation_range=float(direct_translation_range),
            )
        elif self.registration_head_type == "cross_attention_svd":
            self.head = CrossAttentionSVDHead(
                feature_dim=feat_dim,
                num_heads=int(attention_heads),
                dropout=float(attention_dropout),
                match_temperature=match_temperature,
                target_saliency_prior_weight=float(
                    target_saliency_prior_weight
                ),
                translation_residual_range=float(
                    translation_residual_range
                ),
                translation_residual_detach_features=bool(
                    translation_residual_detach_features
                ),
                translation_residual_hidden_dim=int(
                    translation_residual_hidden_dim
                ),
            )
        elif self.registration_head_type == "svd":
            self.head = SVDHead(
                in_dim_src=feat_dim,
                in_dim_tgt=feat_dim,
                robust_matching=robust_matching,
                match_temperature=match_temperature,
            )
        else:
            raise ValueError(f"Unknown registration_head: {self.registration_head_type}")

    def forward(
        self,#源点云，通常是 STL 口内扫描
        p_src,#目标点云，通常是 CBCT 表面
        p_tgt,
        return_aux=False,
        expected_determinant=None,
        source_center_normalized=None,
    ):
        # --- Input and format conversion (no change) ---
        B, N_s, _ = p_src.shape
        _, N_t, _ = p_tgt.shape
#这一步只是把不同样本的点拼接起来，没有混合点的坐标。
        p_src_flat = p_src.view(-1, 3)
        batch_src_orig = torch.arange(B, device=p_src.device).repeat_interleave(N_s)
#目标点云执行相同操作
        p_tgt_flat = p_tgt.view(-1, 3)
        batch_tgt_orig = torch.arange(B, device=p_tgt.device).repeat_interleave(N_t)

        # 双分支 PointNet++ 编码
        #STL 是高分辨率口扫表面；
        #CBCT 是阈值提取的骨表面；
#f_*_sampled	降采样点的局部几何特征
# p_*_sampled	降采样后保留的点坐标
# batch_*_sampled	每个降采样点所属的病例编号
# g_*	每个病例的全局平均特征
        f_src_sampled, p_src_sampled, batch_src_sampled, g_src = self.src_encoder(p_src_flat, batch_src_orig)
        f_tgt_sampled, p_tgt_sampled, batch_tgt_sampled, g_tgt = self.tgt_encoder(p_tgt_flat, batch_tgt_orig)

        # --- Reshape using correct batch indices ---
        # f_src_sampled: [B * N_s_sampled, C]
        # batch_src_sampled: [B * N_s_sampled]
        # Now their dimensions match!
        #源特征稠密化
        f_src, _ = to_dense_batch(f_src_sampled, batch_src_sampled, fill_value=0.0, max_num_nodes=512)
        p_src_ds, _ = to_dense_batch(p_src_sampled, batch_src_sampled, fill_value=0.0, max_num_nodes=512)

        # Target starts with twice as many points (8192 by default), therefore
        # the encoder produces about 1024 points here.  Keep this configurable:
        # 512 reproduces old checkpoints, while 1024 gives newly trained models
        # full CBCT coverage.
        #目标特征稠密化
        f_tgt, _ = to_dense_batch(
            f_tgt_sampled,
            batch_tgt_sampled,
            fill_value=0.0,
            max_num_nodes=self.target_dense_points,
        )
        p_tgt_ds, _ = to_dense_batch(
            p_tgt_sampled,
            batch_tgt_sampled,
            fill_value=0.0,
            max_num_nodes=self.target_dense_points,
        )

        # --- Dimension check ---
        # f_src: [B, 512, C]
        # p_src_ds: [B, 512, 3]
        # f_tgt: [B, 512, C]
        # p_tgt_ds: [B, 512, 3]
        # Dimensions are now correct and dense tensors are ready for use

        # --- SVD Head (input remains unchanged) ---
# 双向 Cross Attention
# → 双 Softmax 匹配
# → 加权 SVD
# → correspondence_transform
        if self.registration_head_type == "hybrid_direct":
            outputs = self.head(p_src_ds, f_src, p_tgt_ds, f_tgt, g_src, g_tgt)
            return outputs if return_aux else outputs["transform"]

        if self.registration_head_type == "cross_attention_svd":
            outputs = self.head(
                p_src_ds,
                f_src,
                p_tgt_ds,
                f_tgt,
                g_src=g_src,
                g_tgt=g_tgt,
                source_center_normalized=source_center_normalized,
                expected_determinant=expected_determinant,
            )
            outputs["source_points"] = p_src_ds
            outputs["target_points"] = p_tgt_ds
            return outputs if return_aux else outputs["transform"]

        transform_pred = self.head(p_src_ds, f_src, p_tgt_ds, f_tgt)
        return transform_pred

    def score_target_candidates(self, p_tgt):
        """Encode a coarse CBCT candidate cloud and score jaw saliency."""
        if self.registration_head_type != "cross_attention_svd":
            raise RuntimeError(
                "Target saliency scoring requires cross_attention_svd"
            )
        batch_size, num_points, _ = p_tgt.shape
        p_tgt_flat = p_tgt.reshape(-1, 3)
        batch = torch.arange(
            batch_size, device=p_tgt.device
        ).repeat_interleave(num_points)
        features, positions, sampled_batch, _ = self.tgt_encoder(
            p_tgt_flat, batch
        )
        logits = self.head.score_target_points(positions, features)
        return positions, logits, sampled_batch
