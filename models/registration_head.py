# import torch
# import torch.nn as nn

# class SVDHead(nn.Module):
#     def __init__(self, in_dim_src, in_dim_tgt):
#         super().__init__()
#         # Simple attention mechanism
#         self.attention = nn.Linear(in_dim_src, in_dim_tgt)

#     def forward(self, p_src, f_src, p_tgt, f_tgt):
#         # p_src: (B, N_s, 3), f_src: (B, N_s, C)
#         # p_tgt: (B, N_t, 3), f_tgt: (B, N_t, C)

#         # Compute soft correspondence
#         f_src_att = self.attention(f_src)  # (B, N_s, C_t)
#         # bmm: batch matrix multiplication
#         affinity = torch.bmm(f_src_att, f_tgt.transpose(1, 2))  # (B, N_s, N_t)
#         prob = torch.softmax(affinity, dim=2)

#         # Compute expected corresponding points
#         p_corr = torch.bmm(prob, p_tgt)  # (B, N_s, 3)

#         # Compute weighted centroids
#         p_src_centroid = torch.mean(p_src, dim=1, keepdim=True)
#         p_corr_centroid = torch.mean(p_corr, dim=1, keepdim=True)

#         p_src_centered = p_src - p_src_centroid
#         p_corr_centered = p_corr - p_corr_centroid

#         # Build cross-covariance matrix
#         H = torch.bmm(p_src_centered.transpose(1, 2), p_corr_centered)  # (B, 3, 3)

#         # SVD decomposition
#         try:
#             U, _, V = torch.svd(H, some=False, compute_uv=True)
#         except torch.linalg.LinAlgError:
#             # SVD may not converge, return identity matrix
#             identity = torch.eye(3).to(H.device).unsqueeze(0).repeat(H.shape[0], 1, 1)
#             return identity, torch.zeros(H.shape[0], 3).to(H.device)

#         R = torch.bmm(V, U.transpose(1, 2))

#         # Fix possible reflections
#         det = torch.det(R)
#         diag = torch.tensor([1.0, 1.0, -1.0], device=R.device)
#         V_prime = V * diag
#         R_det_neg = torch.bmm(V_prime, U.transpose(1, 2))
#         R = torch.where(det.view(-1, 1, 1) < 0, R_det_neg, R)

#         t = p_corr_centroid.squeeze(1) - torch.bmm(R, p_src_centroid.transpose(1, 2)).squeeze(-1)

#         # Assemble 4x4 transformation matrix
#         transform = torch.eye(4, device=R.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
#         transform[:, :3, :3] = R
#         transform[:, :3, 3] = t

#         return transform

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_rigid_transform(source, target, weights, expected_determinant=None):
    """Differentiable weighted orthogonal Procrustes alignment.

    Task2 includes valid left-handed transforms (determinant -1), so the
    solution must cover O(3), not only proper rotations in SO(3).
    """
    # SVD is intentionally FP32 even when the surrounding network uses BF16.
    # CUDA SVD does not support every low-precision dtype and pose accuracy is
    # sensitive to small covariance errors.
    with torch.autocast(device_type=source.device.type, enabled=False):
        source = source.float()
        target = target.float()
        weights = weights.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights_3d = weights.unsqueeze(-1)
        source_center = torch.sum(weights_3d * source, dim=1, keepdim=True)
        target_center = torch.sum(weights_3d * target, dim=1, keepdim=True)
        source_centered = source - source_center
        target_centered = target - target_center
        covariance = torch.bmm(
            (weights_3d * source_centered).transpose(1, 2), target_centered
        )

        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        # The unconstrained V @ U^T solution is optimal over O(3). Do not
        # force determinant +1: that made an entire CBCT domain impossible.
        rotation = torch.bmm(vh.transpose(1, 2), u.transpose(1, 2))
        if expected_determinant is not None:
            expected_determinant = torch.where(
                expected_determinant.to(
                    device=rotation.device, dtype=rotation.dtype
                ).view(-1) < 0.0,
                -torch.ones(rotation.shape[0], device=rotation.device),
                torch.ones(rotation.shape[0], device=rotation.device),
            )
            raw_determinant = torch.det(rotation)
            correction = torch.ones(
                (rotation.shape[0], 3),
                device=rotation.device,
                dtype=rotation.dtype,
            )
            correction[:, -1] = expected_determinant * raw_determinant
            rotation = torch.bmm(
                vh.transpose(1, 2) * correction.unsqueeze(1),
                u.transpose(1, 2),
            )
        translation = target_center.squeeze(1) - torch.bmm(
            rotation, source_center.transpose(1, 2)
        ).squeeze(-1)

    transform = torch.eye(
        4, device=rotation.device, dtype=rotation.dtype
    ).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    return transform


def rotation_6d_to_matrix(rotation_6d):
    """Continuous 6-D rotation representation from Zhou et al."""
    first = F.normalize(rotation_6d[:, 0:3], dim=1, eps=1e-6)
    second_raw = rotation_6d[:, 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(dim=1, keepdim=True) * first,
        dim=1,
        eps=1e-6,
    )
    third = torch.cross(first, second, dim=1)
    return torch.stack([first, second, third], dim=2)


class CrossAttentionSVDHead(nn.Module):
    """Cross-modal SVD rotation with a bounded translation correction.

    Rotation is always recovered from learned correspondences by weighted
    Procrustes.  A small global branch may correct only the SVD translation;
    it cannot change rotation or handedness.
    """

    def __init__(
        self,
        feature_dim,
        num_heads=4,
        dropout=0.1,
        match_temperature=0.07,
        target_saliency_prior_weight=1.0,
        translation_residual_range=0.0,
        translation_residual_detach_features=True,
        translation_residual_hidden_dim=32,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.match_temperature = float(match_temperature)
        self.target_saliency_prior_weight = float(target_saliency_prior_weight)
        self.translation_residual_range = max(
            0.0, float(translation_residual_range)
        )
        self.translation_residual_detach_features = bool(
            translation_residual_detach_features
        )
        self.source_cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.target_cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.source_norm = nn.LayerNorm(feature_dim)
        self.target_norm = nn.LayerNorm(feature_dim)
        self.source_projection = nn.Linear(feature_dim, feature_dim)
        self.target_projection = nn.Linear(feature_dim, feature_dim)
        translation_hidden = max(8, int(translation_residual_hidden_dim))
        self.translation_residual_head = nn.Sequential(
            nn.Linear(3 * feature_dim + 15, translation_hidden),
            nn.LayerNorm(translation_hidden),
            nn.GELU(),
            nn.Linear(translation_hidden, 3),
        )
        # Start from the exact SVD solution. The residual branch must earn any
        # correction from supervised translation gradients.
        nn.init.zeros_(self.translation_residual_head[-1].weight)
        nn.init.zeros_(self.translation_residual_head[-1].bias)
        saliency_hidden = max(64, feature_dim // 2)
        self.target_saliency_head = nn.Sequential(
            nn.Linear(feature_dim + 3, saliency_hidden),
            nn.LayerNorm(saliency_hidden),
            nn.GELU(),
            nn.Linear(saliency_hidden, saliency_hidden),
            nn.GELU(),
            nn.Linear(saliency_hidden, 1),
        )
#给 CBCT 目标点云中的每个点打分，判断这个点属于"上颌骨/下颌骨"区域的可能性
    def score_target_points(self, p_tgt, f_tgt):
        return self.target_saliency_head(
            torch.cat([f_tgt, p_tgt], dim=-1)
        ).squeeze(-1)

    def forward(
        self,
        p_src,
        f_src,
        p_tgt,
        f_tgt,
        g_src=None,
        g_tgt=None,
        source_center_normalized=None,
        expected_determinant=None,
    ):
#f_*_sampled	降采样点的局部几何特征
# p_*_sampled	降采样后保留的点坐标
# batch_*_sampled	每个降采样点所属的病例编号
# g_*	每个病例的全局平均特征
        # Score saliency from target-only encoder features so the same head can
        # be used before STL matching during coarse inference sampling.
        # Keep saliency supervision independent from registration features.
        # The scorer adapts to encoder features, but its auxiliary loss cannot
        # pull the target encoder away from the pose objective.
        target_saliency_logits = self.score_target_points(
            p_tgt, f_tgt.detach()
        )
        source_context, _ = self.source_cross_attention(
            f_src, f_tgt, f_tgt, need_weights=False
        )
        target_context, _ = self.target_cross_attention(
            f_tgt, f_src, f_src, need_weights=False
        )
        f_src = self.source_norm(f_src + source_context)
        f_tgt = self.target_norm(f_tgt + target_context)

        #这是把 STL 和 CBCT 的特征投影到同一个向量空间，然后归一化，方便后续计算点之间的匹配相似度。
        source_match = F.normalize(self.source_projection(f_src), dim=2)
        target_match = F.normalize(self.target_projection(f_tgt), dim=2)
        affinity = torch.bmm(source_match, target_match.transpose(1, 2))
        affinity = affinity / max(self.match_temperature, 1e-4)
        if self.target_saliency_prior_weight != 0.0:
            affinity = affinity + self.target_saliency_prior_weight * F.logsigmoid(
                target_saliency_logits
            ).unsqueeze(1)

        row_probability = torch.softmax(affinity, dim=2)
        column_probability = torch.softmax(affinity, dim=1)
        joint_probability = row_probability * column_probability
        joint_probability = joint_probability / joint_probability.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-8)
        corresponding_target = torch.bmm(joint_probability, p_tgt)

        peak = joint_probability.max(dim=2).values
        entropy = -torch.sum(
            joint_probability * torch.log(joint_probability.clamp_min(1e-8)),
            dim=2,
        )
        confidence = peak * torch.exp(-entropy)

        #用“加权的 SVD 算法”，根据源点云和它对应的目标点云位置，
        #计算出从 STL 到 CBCT 的初始刚性变换矩阵（旋转 + 平移）
        base_transform = weighted_rigid_transform(
            p_src,
            corresponding_target,
            confidence,
            expected_determinant=expected_determinant,
        )
        if g_src is None:
            g_src = f_src.mean(dim=1)
        if g_tgt is None:
            g_tgt = f_tgt.mean(dim=1)
        base_translation = base_transform[:, :3, 3]
        g_src = g_src.to(
            device=base_translation.device, dtype=base_translation.dtype
        )
        g_tgt = g_tgt.to(
            device=base_translation.device, dtype=base_translation.dtype
        )
        if source_center_normalized is None:
            source_center_normalized = torch.zeros_like(base_translation)
        source_center_normalized = source_center_normalized.to(
            device=base_translation.device,
            dtype=base_translation.dtype,
        )
        residual_features = torch.cat(
            [
                g_src,
                g_tgt,
                g_tgt - g_src,
                base_translation,
                base_transform[:, :3, :3].reshape(-1, 9),
                source_center_normalized,
            ],
            dim=1,
        )
        if self.translation_residual_detach_features:
            residual_features = residual_features.detach()
        residual_raw = self.translation_residual_head(residual_features)
        translation_residual = (
            self.translation_residual_range * torch.tanh(residual_raw)
        )
        # tanh bounds each coordinate; additionally bound the full vector so
        # ``translation_residual_range`` has a geometric norm interpretation.
        residual_norm = torch.linalg.norm(
            translation_residual, dim=1, keepdim=True
        ).clamp_min(1e-8)
        norm_scale = torch.clamp(
            self.translation_residual_range / residual_norm,
            max=1.0,
        )
        translation_residual = translation_residual * norm_scale

        transform = base_transform.clone()
        transform[:, :3, 3] = base_translation + translation_residual
        return {
            "transform": transform,
            "base_transform": base_transform,
            "translation_residual": translation_residual,
            "match_logits": affinity,
            "target_saliency_logits": target_saliency_logits,
        }


class HybridDirectRegistrationHead(nn.Module):
    """Single-stage global-pose and correspondence fusion head.

    Both branches see the original unregistered clouds and predict the final
    pose.  The global branch handles large displacements while symmetric
    cross-attention and dual-softmax correspondences preserve local geometry.
    """

    def __init__(
        self,
        feature_dim,
        num_heads=4,
        dropout=0.1,
        match_temperature=0.07,
        translation_range=2.0,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.match_temperature = float(match_temperature)
        self.translation_range = float(translation_range)

        self.source_cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.target_cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.source_norm = nn.LayerNorm(feature_dim)
        self.target_norm = nn.LayerNorm(feature_dim)
        self.source_projection = nn.Linear(feature_dim, feature_dim)
        self.target_projection = nn.Linear(feature_dim, feature_dim)

        hidden_dim = max(128, feature_dim * 2)
        self.global_pose = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),
        )
        nn.init.normal_(self.global_pose[-1].weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            self.global_pose[-1].bias.copy_(
                torch.tensor(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                )
            )
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
        )
        # Begin balanced; branch supervision teaches which hypothesis to trust.
        nn.init.constant_(self.fusion_gate[-1].bias, 0.0)

    def forward(self, p_src, f_src, p_tgt, f_tgt, g_src, g_tgt):
        source_context, _ = self.source_cross_attention(
            f_src, f_tgt, f_tgt, need_weights=False
        )
        target_context, _ = self.target_cross_attention(
            f_tgt, f_src, f_src, need_weights=False
        )
        f_src = self.source_norm(f_src + source_context)
        f_tgt = self.target_norm(f_tgt + target_context)

        source_match = F.normalize(self.source_projection(f_src), dim=2)
        target_match = F.normalize(self.target_projection(f_tgt), dim=2)
        affinity = torch.bmm(source_match, target_match.transpose(1, 2))
        affinity = affinity / max(self.match_temperature, 1e-4)

        # Dual softmax suppresses one-to-many matches common in repeated teeth.
        row_probability = torch.softmax(affinity, dim=2)
        column_probability = torch.softmax(affinity, dim=1)
        joint_probability = row_probability * column_probability
        joint_probability = joint_probability / joint_probability.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-8)
        corresponding_target = torch.bmm(joint_probability, p_tgt)

        peak = joint_probability.max(dim=2).values
        entropy = -torch.sum(
            joint_probability * torch.log(joint_probability.clamp_min(1e-8)), dim=2
        )
        entropy_confidence = torch.exp(-entropy)
        confidence = peak * entropy_confidence
        correspondence_transform = weighted_rigid_transform(
            p_src, corresponding_target, confidence
        )

        global_input = torch.cat(
            [g_src, g_tgt, torch.abs(g_src - g_tgt), g_src * g_tgt], dim=1
        )
        global_parameters = self.global_pose(global_input)
        global_rotation = rotation_6d_to_matrix(global_parameters[:, :6])
        global_translation = self.translation_range * torch.tanh(
            global_parameters[:, 6:9]
        )
        global_transform = torch.eye(
            4, device=p_src.device, dtype=p_src.dtype
        ).unsqueeze(0).repeat(p_src.shape[0], 1, 1)
        global_transform[:, :3, :3] = global_rotation
        global_transform[:, :3, 3] = global_translation

        global_weight = torch.sigmoid(self.fusion_gate(global_input)).view(-1, 1, 1)
        # Blend the first two rotation columns in the continuous 6-D space.
        # This avoids a second SVD and its unstable gradients near repeated
        # singular values.
        correspondence_6d = torch.cat(
            [
                correspondence_transform[:, :3, 0],
                correspondence_transform[:, :3, 1],
            ],
            dim=1,
        )
        global_6d = torch.cat(
            [global_transform[:, :3, 0], global_transform[:, :3, 1]], dim=1
        )
        fused_rotation = rotation_6d_to_matrix(
            (1.0 - global_weight.squeeze(-1)) * correspondence_6d
            + global_weight.squeeze(-1) * global_6d
        )
        translation_weight = global_weight.squeeze(-1)
        fused_translation = (
            (1.0 - translation_weight) * correspondence_transform[:, :3, 3]
            + translation_weight * global_transform[:, :3, 3]
        )
        transform = torch.eye(
            4, device=p_src.device, dtype=p_src.dtype
        ).unsqueeze(0).repeat(p_src.shape[0], 1, 1)
        transform[:, :3, :3] = fused_rotation
        transform[:, :3, 3] = fused_translation
        return {
            "transform": transform,
            "correspondence_transform": correspondence_transform,
            "global_transform": global_transform,
            "fusion_gate": global_weight.view(-1),
        }

class SVDHead(nn.Module):
    def __init__(self, in_dim_src, in_dim_tgt, robust_matching=False, match_temperature=0.1):
        super().__init__()
        # Simple attention mechanism
        self.attention = nn.Linear(in_dim_src, in_dim_tgt)
        self.robust_matching = bool(robust_matching)
        self.match_temperature = float(match_temperature)

    def forward(self, p_src, f_src, p_tgt, f_tgt):
        # p_src: (B, N_s, 3), f_src: (B, N_s, C)
        # p_tgt: (B, N_t, 3), f_tgt: (B, N_t, C)

        # Compute soft correspondence
        f_src_att = self.attention(f_src)  # (B, N_s, C_t)
        # bmm: batch matrix multiplication
        if self.robust_matching:
            f_src_match = F.normalize(f_src_att, dim=2)
            f_tgt_match = F.normalize(f_tgt, dim=2)
            affinity = torch.bmm(f_src_match, f_tgt_match.transpose(1, 2))
            affinity = affinity / max(self.match_temperature, 1e-4)
        else:
            affinity = torch.bmm(f_src_att, f_tgt.transpose(1, 2))
        prob = torch.softmax(affinity, dim=2)

        # Compute expected corresponding points
        p_corr = torch.bmm(prob, p_tgt)  # (B, N_s, 3)

        # Down-weight ambiguous correspondences.  The old implementation gave
        # low-confidence source points the same influence as sharp matches.
        if self.robust_matching:
            confidence = prob.max(dim=2).values
            confidence = confidence / confidence.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            confidence = torch.full_like(prob[:, :, 0], 1.0 / p_src.shape[1])

        # Compute confidence-weighted centroids
        weights = confidence.unsqueeze(-1)
        p_src_centroid = torch.sum(weights * p_src, dim=1, keepdim=True)
        p_corr_centroid = torch.sum(weights * p_corr, dim=1, keepdim=True)

        p_src_centered = p_src - p_src_centroid
        p_corr_centered = p_corr - p_corr_centroid

        # Build cross-covariance matrix
        H = torch.bmm((weights * p_src_centered).transpose(1, 2), p_corr_centered)  # (B, 3, 3)

        # SVD decomposition
        try:
            U, _, V = torch.svd(H, some=False, compute_uv=True)
        except torch.linalg.LinAlgError:
            # SVD may not converge, return identity matrix
            identity = torch.eye(3).to(H.device).unsqueeze(0).repeat(H.shape[0], 1, 1)
            return identity, torch.zeros(H.shape[0], 3).to(H.device)

        R = torch.bmm(V, U.transpose(1, 2))

        # Keep the O(3) solution. Valid challenge transforms may have det=-1.

        t = p_corr_centroid.squeeze(1) - torch.bmm(R, p_src_centroid.transpose(1, 2)).squeeze(-1)

        # Assemble 4x4 transformation matrix
        transform = torch.eye(4, device=R.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t

        return transform
