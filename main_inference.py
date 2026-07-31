import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import yaml
from tqdm import tqdm
import trimesh

from data.dataset import (
    compute_aabb,
    center_align_points,
    crop_cbct_points_by_relative_z,
    load_surface_mesh,
    resolve_cbct_roi_z_bounds,
    resolve_cbct_scale_mode,
    sample_surface_points,
    stable_seed,
)
from models.main_model import RegistrationModel
from utils.domain_correction import affine_signature, load_domain_corrections
from utils.transform_utils import recover_original_transform


def sample_points(points, num_points, rng=None):
    if points.shape[0] <= num_points:
        return points
    chooser = rng if rng is not None else np.random
    indices = chooser.choice(points.shape[0], num_points, replace=False)
    return points[indices]


def build_target_saliency_pool(
    model,
    device,
    target_centered,
    full_scale,
    scale_mode,
    config,
    rng,
):
    """Score a coarse CBCT pool and return high-confidence jaw points."""
    candidate_count = min(
        target_centered.shape[0],
        max(
            int(config['num_points_cbct']),
            int(config.get('target_saliency_candidate_points', 65536)),
        ),
    )
    candidate_indices = rng.choice(
        target_centered.shape[0], candidate_count, replace=False
    )
    candidates = target_centered[candidate_indices]
    if scale_mode == "full":
        saliency_scale = full_scale
    else:
        saliency_scale = np.max(np.linalg.norm(candidates, axis=1))
    if saliency_scale < 1e-6:
        saliency_scale = 1.0

    candidate_tensor = torch.from_numpy(
        (candidates / saliency_scale).astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        positions, logits, sampled_batch = model.score_target_candidates(
            candidate_tensor
        )
    if not bool((sampled_batch == 0).all()):
        raise RuntimeError("Unexpected multi-batch saliency output")
    positions = positions.float().cpu().numpy() * saliency_scale
    scores = logits.float().cpu().numpy()
    top_pool_count = min(
        positions.shape[0],
        max(
            1,
            int(config.get('target_saliency_top_pool_points', 6144)),
        ),
    )
    if top_pool_count < positions.shape[0]:
        top_indices = np.argpartition(scores, -top_pool_count)[-top_pool_count:]
    else:
        top_indices = np.arange(positions.shape[0])
    pool = positions[top_indices]
    selected_scores = scores[top_indices]
    return pool, {
        "candidate_points": candidate_count,
        "encoded_points": int(positions.shape[0]),
        "pool_points": int(pool.shape[0]),
        "score_mean": float(scores.mean()),
        "pool_score_min": float(selected_scores.min()),
        "pool_score_mean": float(selected_scores.mean()),
    }


def apply_transform(points, transform):
    points_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
    return (points_h @ transform.T)[:, :3]


def rigid_delta_magnitude(candidate, reference):
    delta = candidate @ np.linalg.inv(reference)
    translation_mm = float(np.linalg.norm(candidate[:3, 3] - reference[:3, 3]))
    if np.linalg.det(delta[:3, :3]) < 0.0:
        rotation_deg = 180.0
    else:
        cos_angle = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = float(np.degrees(np.arccos(cos_angle)))
    return translation_mm, rotation_deg


def transform_distance(first, second, translation_scale=5.0):
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative_rotation = first[:3, :3] @ second[:3, :3].T
    if np.linalg.det(relative_rotation) < 0.0:
        # A handedness change is not a small TTA perturbation. Treat the two
        # O(3) components as distinct consensus modes.
        rotation = 180.0
    else:
        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation = float(np.degrees(np.arccos(cosine)))
    return translation, rotation, rotation + translation / max(translation_scale, 1e-6)


def aggregate_transforms(
    transforms,
    max_translation_spread=20.0,
    max_rotation_spread=15.0,
):
    """Robustly average a consensus subset of TTA poses on O(3)."""
    if len(transforms) == 1:
        return transforms[0].astype(np.float32), 1

    pair_scores = np.zeros((len(transforms), len(transforms)), dtype=np.float32)
    for row in range(len(transforms)):
        for col in range(row + 1, len(transforms)):
            _, _, score = transform_distance(transforms[row], transforms[col])
            pair_scores[row, col] = score
            pair_scores[col, row] = score
    medoid_index = int(np.argmin(pair_scores.sum(axis=1)))
    medoid = transforms[medoid_index]
    inliers = []
    for transform in transforms:
        translation, rotation, _ = transform_distance(transform, medoid)
        if translation <= max_translation_spread and rotation <= max_rotation_spread:
            inliers.append(transform)
    if not inliers:
        inliers = [medoid]

    # scipy Rotation rejects det=-1 matrices. Average in matrix space and
    # project back to the medoid's connected component of O(3).
    mean_matrix = np.mean(np.stack([t[:3, :3] for t in inliers]), axis=0)
    u, _, vt = np.linalg.svd(mean_matrix)
    mean_orthogonal = u @ vt
    desired_sign = -1.0 if np.linalg.det(medoid[:3, :3]) < 0.0 else 1.0
    if np.linalg.det(mean_orthogonal) * desired_sign < 0.0:
        u[:, -1] *= -1.0
        mean_orthogonal = u @ vt
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = mean_orthogonal.astype(np.float32)
    result[:3, 3] = np.median(
        np.stack([t[:3, 3] for t in inliers]), axis=0
    ).astype(np.float32)
    return result, len(inliers)


def aggregate_translations(transforms, max_translation_spread=20.0):
    """Robust median of TTA translations, independent of rotation quality."""
    translations = np.stack([t[:3, 3] for t in transforms])
    if translations.shape[0] == 1:
        return translations[0].astype(np.float32), 1
    distances = np.linalg.norm(
        translations[:, None, :] - translations[None, :, :], axis=2
    )
    medoid_index = int(np.argmin(distances.sum(axis=1)))
    inlier_mask = distances[medoid_index] <= float(max_translation_spread)
    if not np.any(inlier_mask):
        inlier_mask[medoid_index] = True
    translation = np.median(translations[inlier_mask], axis=0)
    return translation.astype(np.float32), int(np.count_nonzero(inlier_mask))


def constrain_lower_translation(upper_transform, lower_transform, config):
    """Use the paired upper jaw only as a gated lower-translation safeguard."""
    result = lower_transform.copy()
    prior = np.asarray(
        config.get("lower_minus_upper_translation_prior_mm", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    if prior.shape != (3,):
        raise ValueError("lower_minus_upper_translation_prior_mm must have 3 values")
    upper_anchor = upper_transform[:3, 3] + prior
    disagreement = float(np.linalg.norm(lower_transform[:3, 3] - upper_anchor))
    threshold = float(config.get("lower_translation_constraint_threshold_mm", 12.0))
    blend = float(config.get("lower_translation_upper_blend", 0.5))
    applied = disagreement > threshold and blend > 0.0
    if applied:
        blend = float(np.clip(blend, 0.0, 1.0))
        result[:3, 3] = (
            (1.0 - blend) * lower_transform[:3, 3]
            + blend * upper_anchor
        )
    return result, disagreement, applied


def crop_target_around_source(src_points, tgt_points, transform, margin_mm):
    transformed_src = apply_transform(src_points, transform)
    bbox_min = np.min(transformed_src, axis=0) - margin_mm
    bbox_max = np.max(transformed_src, axis=0) + margin_mm
    keep = np.all((tgt_points >= bbox_min) & (tgt_points <= bbox_max), axis=1)
    return tgt_points[keep]


def trimmed_nn_error(src_points, tgt_tree, transform, trim_ratio):
    distances, _ = tgt_tree.query(apply_transform(src_points, transform))
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return None
    if 0.0 < trim_ratio < 1.0:
        keep_count = max(3, int(distances.size * trim_ratio))
        distances = np.partition(distances, keep_count - 1)[:keep_count]
    return float(np.mean(distances))


def estimate_rigid_transform(src_points, dst_points):
    src_center = np.mean(src_points, axis=0)
    dst_center = np.mean(dst_points, axis=0)

    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center

    cov = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(cov)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    translation = dst_center - rotation @ src_center

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = translation.astype(np.float32)
    return transform


def extract_cbct_points(cbct_img, threshold, surface_only=False):
    cbct_data = cbct_img.get_fdata()
    mask = cbct_data > threshold
    if surface_only:
        eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
        mask = mask & ~eroded

    coords = np.argwhere(mask)
    if coords.shape[0] == 0 and surface_only:
        coords = np.argwhere(cbct_data > threshold)
    if coords.shape[0] == 0:
        raise ValueError(f"No CBCT points found above threshold {threshold}")

    coords_h = np.hstack([coords, np.ones((coords.shape[0], 1), dtype=np.float32)])
    return (cbct_img.affine @ coords_h.T).T[:, :3].astype(np.float32)


def refine_transform_with_icp(
    src_points,
    tgt_points,
    init_transform,
    max_iterations=25,
    trim_ratio=0.7,
    tolerance=1e-3,
    max_correspondence_distance=20.0,
    max_total_translation=10.0,
    max_total_rotation=8.0,
    acceptance_target_points=None,
):
    if src_points.shape[0] < 3 or tgt_points.shape[0] < 3:
        return init_transform, {
            "iterations": 0,
            "mean_error_before": None,
            "mean_error_after": None,
            "accepted": False,
        }

    tree = cKDTree(tgt_points)
    acceptance_tree = tree
    if acceptance_target_points is not None and acceptance_target_points.shape[0] >= 3:
        acceptance_tree = cKDTree(acceptance_target_points)
    transform = init_transform.astype(np.float32).copy()
    initial_transform = transform.copy()
    prev_error = None
    mean_error_before = trimmed_nn_error(
        src_points, acceptance_tree, transform, trim_ratio
    )

    for iteration in range(max_iterations):
        transformed = apply_transform(src_points, transform)
        distances, indices = tree.query(transformed)

        valid_mask = np.isfinite(distances)
        if max_correspondence_distance is not None:
            valid_mask &= distances <= max_correspondence_distance
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size < 3:
            break

        if 0.0 < trim_ratio < 1.0:
            keep_count = max(3, int(valid_indices.size * trim_ratio))
            ranked = valid_indices[np.argsort(distances[valid_indices])[:keep_count]]
        else:
            ranked = valid_indices

        src_subset = src_points[ranked]
        dst_subset = tgt_points[indices[ranked]]
        new_transform = estimate_rigid_transform(src_subset, dst_subset)
        mean_error = float(np.mean(distances[ranked]))

        total_translation, total_rotation = rigid_delta_magnitude(new_transform, initial_transform)
        if total_translation > max_total_translation or total_rotation > max_total_rotation:
            break

        transform = new_transform
        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    mean_error_after = trimmed_nn_error(
        src_points, acceptance_tree, transform, trim_ratio
    )
    accepted = (
        mean_error_before is not None
        and mean_error_after is not None
        and mean_error_after < mean_error_before
    )
    if not accepted:
        transform = initial_transform
        mean_error_after = mean_error_before

    return transform, {
        "iterations": iteration + 1 if "iteration" in locals() else 0,
        "mean_error_before": mean_error_before,
        "mean_error_after": mean_error_after,
        "accepted": accepted,
    }


def resolve_jaw_config_value(config, key, jaw_type, default):
    """Resolve a scalar option or a jaw-specific YAML mapping."""
    value = config.get(key, default)
    if isinstance(value, dict):
        if jaw_type not in value:
            raise ValueError(f"{key}.{jaw_type} is required")
        return value[jaw_type]
    return value


def load_model_for_inference(config, jaw_type, checkpoint_filename="best_model.pth"):
    """Load the model for the specified jaw type."""
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = RegistrationModel(
        feat_dim=config['feature_dim'],
        target_dense_points=int(config.get('target_dense_points', 512)),
        robust_matching=bool(config.get('robust_matching', False)),
        match_temperature=float(config.get('match_temperature', 0.1)),
        registration_head=str(config.get('registration_head', 'svd')),
        attention_heads=int(config.get('attention_heads', 4)),
        attention_dropout=float(config.get('attention_dropout', 0.1)),
        direct_translation_range=float(config.get('direct_translation_range', 2.0)),
        target_saliency_prior_weight=float(
            config.get('target_saliency_prior_weight', 1.0)
        ),
        translation_residual_range=float(
            resolve_jaw_config_value(
                config, 'translation_residual_range', jaw_type, 0.0
            )
        ),
        translation_residual_detach_features=bool(
            resolve_jaw_config_value(
                config,
                'translation_residual_detach_features',
                jaw_type,
                True,
            )
        ),
        translation_residual_hidden_dim=int(
            config.get('translation_residual_hidden_dim', 32)
        ),
    ).to(device)

    explicit_checkpoint = config.get(f'{jaw_type}_checkpoint_path')
    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint)
        checkpoint_dir = checkpoint_path.parent
    else:
        exp_name = config[f'{jaw_type}_jaw_experiment_name']
        checkpoint_dir = Path(config['output_dir']) / exp_name / "checkpoints"
        checkpoint_path = checkpoint_dir / checkpoint_filename
        if not checkpoint_path.exists():
            fallback_path = checkpoint_dir / "best_model.pth"
            if checkpoint_filename != "best_model.pth" and fallback_path.exists():
                print(
                    f"Warning: {checkpoint_filename} missing for {jaw_type}; "
                    "falling back to best_model.pth"
                )
                checkpoint_path = fallback_path
            else:
                checkpoint_path = checkpoint_dir / "latest_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model weights not found for {jaw_type} jaw: {checkpoint_dir}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded {jaw_type} jaw weights from: {checkpoint_path}")
    print(
        f"  -> {jaw_type} translation residual range: "
        f"{model.head.translation_residual_range:.3f}"
    )
    return model, device


def predict_single_case(
    model,
    device,
    config,
    stl_path,
    cbct_path,
    jaw_type,
    domain_corrections,
    translation_model=None,
):
    """Perform inference on a single STL and CBCT pair, return the final transformation matrix."""
    mesh = load_surface_mesh(stl_path)
    _, _, stl_center = compute_aabb(mesh.vertices)

    cbct_img = nib.load(str(cbct_path))
    signature = affine_signature(cbct_img.affine)
    negative_signatures = {
        str(value)
        for value in config.get('negative_determinant_affine_signatures', [])
    }
    expected_determinant_value = -1.0 if signature in negative_signatures else 1.0
    expected_determinant = torch.tensor(
        [expected_determinant_value], device=device, dtype=torch.float32
    )
    cbct_threshold = float(config.get('cbct_threshold', 800))
    p_tgt = extract_cbct_points(
        cbct_img,
        cbct_threshold,
        surface_only=bool(config.get('cbct_surface_only', False)),
    )
    cbct_min, cbct_max, cbct_center = compute_aabb(p_tgt)
    full_centered_target = center_align_points(p_tgt, cbct_center)
    full_scale = np.max(np.linalg.norm(full_centered_target, axis=1))
    scale_mode = resolve_cbct_scale_mode(config, jaw_type)
    p_tgt, roi_kept_fraction, roi_applied = crop_cbct_points_by_relative_z(
        p_tgt,
        cbct_min,
        cbct_max,
        resolve_cbct_roi_z_bounds(config, jaw_type),
        min_output_points=int(
            config.get('cbct_roi_min_points', config['num_points_cbct'])
        ),
    )
    if bool(config.get('use_anatomical_cbct_roi', False)):
        if roi_applied:
            print(
                f"    -> Anatomical CBCT ROI ({jaw_type}): "
                f"kept={roi_kept_fraction:.1%}, points={p_tgt.shape[0]}"
            )
        else:
            print(
                f"    -> Anatomical CBCT ROI ({jaw_type}) skipped: "
                "too few eligible points"
            )
    p_tgt_centered = center_align_points(p_tgt, cbct_center)

    tta_samples = max(1, int(config.get('tta_samples', 1)))
    base_seed = int(config.get('inference_seed', 2026))
    saliency_pool = None
    if bool(config.get('use_target_saliency_sampling', False)):
        saliency_rng = np.random.default_rng(
            stable_seed(
                base_seed,
                Path(stl_path).parent.name,
                Path(stl_path).stem,
                "target-saliency",
            )
        )
        try:
            saliency_pool, saliency_info = build_target_saliency_pool(
                model,
                device,
                p_tgt_centered,
                full_scale,
                scale_mode,
                config,
                saliency_rng,
            )
            print(
                "    -> Learned CBCT saliency: "
                f"candidates={saliency_info['candidate_points']}, "
                f"encoded={saliency_info['encoded_points']}, "
                f"pool={saliency_info['pool_points']}, "
                f"score_mean={saliency_info['score_mean']:.3f}, "
                f"pool_mean={saliency_info['pool_score_mean']:.3f}"
            )
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if bool(config.get('target_saliency_required', True)):
                raise
            print(f"    -> Learned CBCT saliency disabled: {exc}")
            saliency_pool = None

    transform_candidates = []
    translation_candidates = []
    for tta_index in range(tta_samples):
        rng = np.random.default_rng(
            stable_seed(base_seed, Path(stl_path).parent.name, Path(stl_path).stem, tta_index)
        )
        p_src = sample_surface_points(mesh, config['num_points_stl'], rng=rng)
        p_src = center_align_points(p_src, stl_center)

        if saliency_pool is not None:
            total_target_points = int(config['num_points_cbct'])
            high_fraction = float(
                config.get('target_saliency_high_fraction', 0.75)
            )
            high_count = min(
                total_target_points,
                max(0, int(round(total_target_points * high_fraction))),
            )
            global_count = total_target_points - high_count
            high_indices = rng.choice(
                saliency_pool.shape[0],
                high_count,
                replace=saliency_pool.shape[0] < high_count,
            )
            global_indices = rng.choice(
                p_tgt_centered.shape[0],
                global_count,
                replace=p_tgt_centered.shape[0] < global_count,
            )
            p_tgt_sampled = np.concatenate(
                [
                    saliency_pool[high_indices],
                    p_tgt_centered[global_indices],
                ],
                axis=0,
            )
            rng.shuffle(p_tgt_sampled)
        else:
            replace = p_tgt_centered.shape[0] < config['num_points_cbct']
            target_indices = rng.choice(
                p_tgt_centered.shape[0],
                config['num_points_cbct'],
                replace=replace,
            )
            p_tgt_sampled = p_tgt_centered[target_indices]

        if scale_mode == "full":
            scale = full_scale
        else:
            scale = np.max(np.linalg.norm(p_tgt_sampled, axis=1))
        if scale < 1e-6:
            scale = 1.0

        p_src_tensor = torch.from_numpy((p_src / scale).astype(np.float32)).unsqueeze(0).to(device)
        p_tgt_tensor = torch.from_numpy((p_tgt_sampled / scale).astype(np.float32)).unsqueeze(0).to(device)
        source_center_normalized = torch.from_numpy(
            (stl_center / scale).astype(np.float32)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            transform_pred_norm = model(
                p_src_tensor,
                p_tgt_tensor,
                expected_determinant=expected_determinant,
                source_center_normalized=source_center_normalized,
            )
            translation_pred_norm = None
            if translation_model is not None:
                translation_pred_norm = translation_model(
                    p_src_tensor,
                    p_tgt_tensor,
                    expected_determinant=expected_determinant,
                    source_center_normalized=source_center_normalized,
                )

        transform_candidates.append(
            recover_original_transform(
                transform_pred_norm.squeeze(0).cpu().numpy(),
                stl_center,
                cbct_center,
                scale,
            )
        )
        if translation_pred_norm is not None:
            translation_candidates.append(
                recover_original_transform(
                    translation_pred_norm.squeeze(0).cpu().numpy(),
                    stl_center,
                    cbct_center,
                    scale,
                )
            )

    transform_world, tta_inliers = aggregate_transforms(
        transform_candidates,
        # When metric-specific checkpoints are used, rotation consensus must
        # not reject a good rotation merely because that checkpoint has a
        # weaker translation component.
        max_translation_spread=(
            float("inf")
            if translation_candidates
            else float(config.get('tta_max_translation_spread_mm', 20.0))
        ),
        max_rotation_spread=float(config.get('tta_max_rotation_spread_deg', 15.0)),
    )
    translation_inliers = None
    if translation_candidates:
        translation_world, translation_inliers = aggregate_translations(
            translation_candidates,
            max_translation_spread=float(
                config.get('tta_max_translation_spread_mm', 20.0)
            ),
        )
        transform_world[:3, 3] = translation_world
    if tta_samples > 1:
        determinant = float(np.linalg.det(transform_world[:3, :3]))
        print(
            f"    -> TTA aggregation: samples={tta_samples}, "
            f"consensus={tta_inliers}, det={determinant:+.0f}, "
            f"expected_det={expected_determinant_value:+.0f}"
            + (
                f", translation_consensus={translation_inliers}"
                if translation_inliers is not None
                else ""
            )
        )
    correction_entry = domain_corrections.get(f"{Path(stl_path).stem}|{signature}")
    if correction_entry is None:
        correction_entry = domain_corrections.get(signature)
    if correction_entry is not None:
        correction = correction_entry["matrix"]
        side = correction_entry["side"]
        if side == "left":
            transform_world = correction @ transform_world
        elif side == "right":
            transform_world = transform_world @ correction
        else:
            transform_world = correction @ transform_world @ np.linalg.inv(correction)

    if bool(config.get('use_icp_refinement', False)):
        icp_rng = np.random.default_rng(
            stable_seed(base_seed, Path(stl_path).parent.name, Path(stl_path).stem, "icp")
        )
        icp_source_points = sample_surface_points(
            mesh, int(config.get('icp_source_points', 12000)), rng=icp_rng
        )
        acceptance_target_points = None
        icp_target_points = extract_cbct_points(
            cbct_img, cbct_threshold, surface_only=True
        )
        icp_target_points = crop_target_around_source(
            icp_source_points,
            icp_target_points,
            transform_world,
            margin_mm=float(config.get('icp_roi_margin_mm', 10.0)),
        )
        icp_target_points = sample_points(
            icp_target_points, int(config.get('icp_target_points', 20000)), rng=icp_rng
        )
        min_target_points = int(config.get('icp_min_target_points', 200))
        if icp_target_points.shape[0] < min_target_points:
            print(
                "    -> ICP skipped: "
                f"ROI target points={icp_target_points.shape[0]} < {min_target_points}"
            )
        else:
            transform_world, icp_info = refine_transform_with_icp(
                icp_source_points.astype(np.float32),
                icp_target_points.astype(np.float32),
                transform_world,
                max_iterations=int(config.get('icp_max_iterations', 25)),
                trim_ratio=float(config.get('icp_trim_ratio', 0.7)),
                tolerance=float(config.get('icp_tolerance', 1e-3)),
                max_correspondence_distance=float(config.get('icp_max_correspondence_distance', 20.0)),
                max_total_translation=float(config.get('icp_max_total_translation_mm', 10.0)),
                max_total_rotation=float(config.get('icp_max_total_rotation_deg', 8.0)),
                acceptance_target_points=acceptance_target_points,
            )
            before = icp_info["mean_error_before"]
            after = icp_info["mean_error_after"]
            print(
                "    -> ICP refinement: "
                f"accepted={icp_info['accepted']}, "
                f"iters={icp_info['iterations']}, "
                f"roi_target_points={icp_target_points.shape[0]}, "
                f"trimmed_nn_before={before if before is not None else 'n/a'}, "
                f"trimmed_nn_after={after if after is not None else 'n/a'}"
            )
    return transform_world


def main():
    parser = argparse.ArgumentParser(description="Run Task2 inference and package a submission.")
    parser.add_argument("--config", default="configs/inference_config.yaml")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    inference_root = Path(config['inference_data_root'])
    submission_zip_path = Path(config['submission_zip_path'])
    prediction_root = Path(config.get('prediction_root', './prediction_output'))
    keep_prediction_root = bool(config.get('keep_prediction_root', True))
    domain_corrections = load_domain_corrections(config.get('domain_corrections_path'))

    if prediction_root.exists():
        shutil.rmtree(prediction_root)
    prediction_root.mkdir(parents=True)

    print("--- Batch inference and packaging started ---")

    metric_specific = bool(config.get('use_metric_specific_checkpoints', False))
    primary_checkpoint = (
        "best_rotation_model.pth" if metric_specific else "best_model.pth"
    )
    print("Loading upper jaw model...")
    upper_model, device = load_model_for_inference(
        config, 'upper', primary_checkpoint
    )
    upper_translation_model = None
    if metric_specific:
        upper_translation_model, _ = load_model_for_inference(
            config, 'upper', "best_translation_model.pth"
        )
    print("Loading lower jaw model...")
    lower_model, _ = load_model_for_inference(
        config, 'lower', primary_checkpoint
    )
    lower_translation_model = None
    if metric_specific:
        lower_translation_model, _ = load_model_for_inference(
            config, 'lower', "best_translation_model.pth"
        )

    case_folders = sorted([p for p in inference_root.iterdir() if p.is_dir()])

    for case_folder in tqdm(case_folders, desc="Processing cases"):
        case_id = case_folder.name
        print(f"\nProcessing case: {case_id}")

        result_case_dir = prediction_root / case_id
        result_case_dir.mkdir()

        cbct_path = case_folder / "CBCT.nii.gz"
        upper_transform = None

        upper_stl_path = case_folder / "upper.stl"
        if upper_stl_path.exists() and cbct_path.exists():
            print("  - Predicting upper jaw...")
            upper_transform = predict_single_case(
                upper_model,
                device,
                config,
                upper_stl_path,
                cbct_path,
                'upper',
                domain_corrections,
                translation_model=upper_translation_model,
            )
            np.save(result_case_dir / "upper_gt.npy", upper_transform)
            print("    -> Upper jaw matrix saved.")
        else:
            print(f"  - Warning: Missing upper jaw or CBCT file for {case_id}, skipping.")

        lower_stl_path = case_folder / "lower.stl"
        if lower_stl_path.exists() and cbct_path.exists():
            print("  - Predicting lower jaw...")
            lower_transform = predict_single_case(
                lower_model,
                device,
                config,
                lower_stl_path,
                cbct_path,
                'lower',
                domain_corrections,
                translation_model=lower_translation_model,
            )
            if (
                upper_transform is not None
                and bool(config.get('use_upper_guided_lower_translation', False))
            ):
                lower_transform, disagreement, constrained = (
                    constrain_lower_translation(
                        upper_transform, lower_transform, config
                    )
                )
                print(
                    "    -> Paired-jaw lower translation: "
                    f"disagreement={disagreement:.3f} mm, "
                    f"constrained={constrained}"
                )
            np.save(result_case_dir / "lower_gt.npy", lower_transform)
            print("    -> Lower jaw matrix saved.")
        else:
            print(f"  - Warning: Missing lower jaw or CBCT file for {case_id}, skipping.")

    print("\nAll cases processed. Creating ZIP archive...")
    shutil.make_archive(
        base_name=str(submission_zip_path.with_suffix('')),
        format='zip',
        root_dir=prediction_root
    )

    print(f"Packaging completed! Submission file saved at: {submission_zip_path}")
    print(f"Prediction folder saved at: {prediction_root}")
    if not keep_prediction_root:
        shutil.rmtree(prediction_root)
        print("Prediction folder cleaned up.")
    print("--- Task completed ---")


if __name__ == '__main__':
    main()
