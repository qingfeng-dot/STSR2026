# # # utils/transform_utils.py

# # import numpy as np
# # import torch


# # def recover_original_coordinates(points, center, scale):
# #     if isinstance(points, torch.Tensor):
# #         points = points.cpu().numpy()
# #     if isinstance(center, torch.Tensor):
# #         center = center.cpu().numpy()

# #     scaled_points = points * scale
# #     original_points = scaled_points + center
# #     return original_points


# # def recover_original_transform(transform, stl_center, cbct_center, scale):
# #     if isinstance(transform, torch.Tensor):
# #         transform = transform.cpu().numpy()
# #     if isinstance(stl_center, torch.Tensor):
# #         stl_center = stl_center.cpu().numpy()
# #     if isinstance(cbct_center, torch.Tensor):
# #         cbct_center = cbct_center.cpu().numpy()
# #     transform = np.asarray(transform, dtype=np.float32)
# #     stl_center = np.asarray(stl_center, dtype=np.float32)
# #     cbct_center = np.asarray(cbct_center, dtype=np.float32)

# #     rotation = transform[:3, :3]
# #     translation_norm = transform[:3, 3]
# #     translation_world = scale * translation_norm + cbct_center - rotation @ stl_center

# #     original_transform = np.eye(4, dtype=np.float32)
# #     original_transform[:3, :3] = rotation
# #     original_transform[:3, 3] = translation_world
# #     return original_transform


# # utils/transform_utils.py

# import numpy as np
# import torch


# def recover_original_coordinates(points, center, scale):
#     if isinstance(points, torch.Tensor):
#         points = points.cpu().numpy()
#     if isinstance(center, torch.Tensor):
#         center = center.cpu().numpy()

#     scaled_points = points * scale
#     original_points = scaled_points + center
#     return original_points


# def recover_original_transform(transform, stl_center, cbct_center, scale):
#     if isinstance(transform, torch.Tensor):
#         transform = transform.cpu().numpy()
#     if isinstance(stl_center, torch.Tensor):
#         stl_center = stl_center.cpu().numpy()
#     if isinstance(cbct_center, torch.Tensor):
#         cbct_center = cbct_center.cpu().numpy()
#     transform = np.asarray(transform, dtype=np.float32)
#     stl_center = np.asarray(stl_center, dtype=np.float32)
#     cbct_center = np.asarray(cbct_center, dtype=np.float32)

#     rotation = transform[:3, :3]
#     translation_norm = transform[:3, 3]
#     translation_world = scale * translation_norm + cbct_center - rotation @ stl_center

#     original_transform = np.eye(4, dtype=np.float32)
#     original_transform[:3, :3] = rotation
#     original_transform[:3, 3] = translation_world
#     return original_transform


# def recover_original_transform_torch(transform, stl_center, cbct_center, scale):
#     if transform.ndim != 3 or transform.shape[1:] != (4, 4):
#         raise ValueError(f"Expected transform with shape [B, 4, 4], got {tuple(transform.shape)}")

#     device = transform.device
#     dtype = transform.dtype
#     stl_center = stl_center.to(device=device, dtype=dtype)
#     cbct_center = cbct_center.to(device=device, dtype=dtype)
#     scale = scale.to(device=device, dtype=dtype).view(-1, 1)

#     rotation = transform[:, :3, :3]
#     translation_norm = transform[:, :3, 3]
#     translation_world = scale * translation_norm + cbct_center - torch.bmm(
#         rotation, stl_center.unsqueeze(-1)
#     ).squeeze(-1)

#     original_transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(transform.shape[0], 1, 1)
#     original_transform[:, :3, :3] = rotation
#     original_transform[:, :3, 3] = translation_world
#     return original_transform

# utils/transform_utils.py

import numpy as np
import torch


def recover_original_coordinates(points, center, scale):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(center, torch.Tensor):
        center = center.cpu().numpy()

    scaled_points = points * scale
    original_points = scaled_points + center
    return original_points


def recover_original_transform(transform, stl_center, cbct_center, scale):
    if isinstance(transform, torch.Tensor):
        transform = transform.cpu().numpy()
    if isinstance(stl_center, torch.Tensor):
        stl_center = stl_center.cpu().numpy()
    if isinstance(cbct_center, torch.Tensor):
        cbct_center = cbct_center.cpu().numpy()
    transform = np.asarray(transform, dtype=np.float32)
    stl_center = np.asarray(stl_center, dtype=np.float32)
    cbct_center = np.asarray(cbct_center, dtype=np.float32)

    rotation = transform[:3, :3]
    translation_norm = transform[:3, 3]
    translation_world = scale * translation_norm + cbct_center - rotation @ stl_center

    original_transform = np.eye(4, dtype=np.float32)
    original_transform[:3, :3] = rotation
    original_transform[:3, 3] = translation_world
    return original_transform


def recover_original_transform_torch(transform, stl_center, cbct_center, scale):
    if transform.ndim != 3 or transform.shape[1:] != (4, 4):
        raise ValueError(f"Expected transform with shape [B, 4, 4], got {tuple(transform.shape)}")

    device = transform.device
    dtype = transform.dtype
    stl_center = stl_center.to(device=device, dtype=dtype)
    cbct_center = cbct_center.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype).view(-1, 1)

    rotation = transform[:, :3, :3]
    translation_norm = transform[:, :3, 3]
    translation_world = scale * translation_norm + cbct_center - torch.bmm(
        rotation, stl_center.unsqueeze(-1)
    ).squeeze(-1)

    original_transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(transform.shape[0], 1, 1)
    original_transform[:, :3, :3] = rotation
    original_transform[:, :3, 3] = translation_world
    return original_transform
