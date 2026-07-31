# # import torch
# # import torch.nn.functional as F


# # def transform_points(points, transform):
# #    points_h = F.pad(points, (0, 1), mode='constant', value=1.0)
# #    transformed_points_h = torch.bmm(points_h, transform.transpose(1, 2))
# #    return transformed_points_h[:, :, :3]


# # def registration_loss(p_src, transform_pred, transform_gt):
# #    p_src_pred = transform_points(p_src, transform_pred)
# #    p_src_gt = transform_points(p_src, transform_gt)

# #    loss = F.mse_loss(p_src_pred, p_src_gt)
# #    return loss


# # [registration_loss.py (line 11)](/D:/github-zip/STSR-Challenge-main/STSR-Challenge-main/STSR-2025/baseline/STSR-Task2-Baseline/losses/registration_loss.py:11)
# # 加了旋转误差、平移误差和组合损失。现在 loss 不再只有 point MSE。
# import torch
# import torch.nn.functional as F


# def transform_points(points, transform):
#     points_h = F.pad(points, (0, 1), mode='constant', value=1.0)
#     transformed_points_h = torch.bmm(points_h, transform.transpose(1, 2))
#     return transformed_points_h[:, :, :3]


# def rotation_error_radians(transform_pred, transform_gt):
#     r_pred = transform_pred[:, :3, :3]
#     r_gt = transform_gt[:, :3, :3]
#     r_rel = torch.bmm(r_pred, r_gt.transpose(1, 2))
#     trace = r_rel[:, 0, 0] + r_rel[:, 1, 1] + r_rel[:, 2, 2]
#     cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
#     return torch.acos(cos_theta)


# def rotation_error_degrees(transform_pred, transform_gt):
#     return torch.rad2deg(rotation_error_radians(transform_pred, transform_gt))


# def translation_error(transform_pred, transform_gt):
#     t_pred = transform_pred[:, :3, 3]
#     t_gt = transform_gt[:, :3, 3]
#     return torch.linalg.norm(t_pred - t_gt, dim=1)


# def registration_loss(
#     p_src,
#     transform_pred,
#     transform_gt,
#     loss_weights=None,
#     return_components=False,
#     metric_transform_pred=None,
#     metric_transform_gt=None,
# ):
#     if loss_weights is None:
#         loss_weights = {"point": 1.0, "rotation": 0.5, "translation": 0.02}

#     point_weight = float(loss_weights.get("point", 1.0))
#     rotation_weight = float(loss_weights.get("rotation", 0.5))
#     translation_weight = float(loss_weights.get("translation", 0.02))

#     p_src_pred = transform_points(p_src, transform_pred)
#     p_src_gt = transform_points(p_src, transform_gt)
#     transform_pred_metric = metric_transform_pred if metric_transform_pred is not None else transform_pred
#     transform_gt_metric = metric_transform_gt if metric_transform_gt is not None else transform_gt

#     point_loss = F.mse_loss(p_src_pred, p_src_gt)
#     rotation_loss = rotation_error_radians(transform_pred_metric, transform_gt_metric).mean()
#     translation_loss = translation_error(transform_pred_metric, transform_gt_metric).mean()

#     loss = (
#         point_weight * point_loss
#         + rotation_weight * rotation_loss
#         + translation_weight * translation_loss
#     )

#     if not return_components:
#         return loss

#     components = {
#         "point_loss": float(point_loss.detach().item()),
#         "rotation_loss_rad": float(rotation_loss.detach().item()),
#         "translation_loss": float(translation_loss.detach().item()),
#     }
#     return loss, components

import torch
import torch.nn.functional as F


def transform_points(points, transform):
    points_h = F.pad(points, (0, 1), mode='constant', value=1.0)
    transformed_points_h = torch.bmm(points_h, transform.transpose(1, 2))
    return transformed_points_h[:, :, :3]


def rotation_error_radians(transform_pred, transform_gt):
    r_pred = transform_pred[:, :3, :3]
    r_gt = transform_gt[:, :3, :3]
    r_rel = torch.bmm(r_pred, r_gt.transpose(1, 2))
    trace = r_rel[:, 0, 0] + r_rel[:, 1, 1] + r_rel[:, 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    # atan2 has useful gradients close to zero and does not impose the old
    # acos clamp floor (~0.081 degrees).
    skew = torch.stack(
        [
            r_rel[:, 2, 1] - r_rel[:, 1, 2],
            r_rel[:, 0, 2] - r_rel[:, 2, 0],
            r_rel[:, 1, 0] - r_rel[:, 0, 1],
        ],
        dim=1,
    )
    sin_theta = 0.5 * torch.linalg.norm(skew, dim=1)
    proper_angle = torch.atan2(sin_theta, cos_theta)

    # The atan2 expression is valid only for a proper relative rotation. If
    # prediction and GT have opposite handedness, follow the challenge's
    # trace/acos metric instead; otherwise a reflection can report zero loss.
    # Avoid an unused acos(±1) branch producing inf*0/NaN in torch.where's
    # backward pass for otherwise correct same-handed predictions.
    metric_cosine = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    metric_angle = torch.acos(metric_cosine)
    same_handedness = torch.det(r_rel) > 0.0
    return torch.where(same_handedness, proper_angle, metric_angle)


def rotation_error_degrees(transform_pred, transform_gt):
    return torch.rad2deg(rotation_error_radians(transform_pred, transform_gt))


def translation_error(transform_pred, transform_gt):
    t_pred = transform_pred[:, :3, 3]
    t_gt = transform_gt[:, :3, 3]
    return torch.linalg.norm(t_pred - t_gt, dim=1)


def registration_loss(
    p_src,
    transform_pred,
    transform_gt,
    loss_weights=None,
    return_components=False,
    metric_transform_pred=None,
    metric_transform_gt=None,
):
    if loss_weights is None:
        loss_weights = {"point": 1.0, "rotation": 0.5, "translation": 0.02}

    point_weight = float(loss_weights.get("point", 1.0))
    rotation_weight = float(loss_weights.get("rotation", 0.5))
    translation_weight = float(loss_weights.get("translation", 0.02))

    p_src_pred = transform_points(p_src, transform_pred)
    p_src_gt = transform_points(p_src, transform_gt)
    transform_pred_metric = metric_transform_pred if metric_transform_pred is not None else transform_pred
    transform_gt_metric = metric_transform_gt if metric_transform_gt is not None else transform_gt

    point_loss = F.mse_loss(p_src_pred, p_src_gt)
    rotation_loss = rotation_error_radians(transform_pred_metric, transform_gt_metric).mean()
    translation_loss = translation_error(transform_pred_metric, transform_gt_metric).mean()

    loss = (
        point_weight * point_loss
        + rotation_weight * rotation_loss
        + translation_weight * translation_loss
    )

    if not return_components:
        return loss

    components = {
        "point_loss": float(point_loss.detach().item()),
        "rotation_loss_rad": float(rotation_loss.detach().item()),
        "translation_loss": float(translation_loss.detach().item()),
    }
    return loss, components
