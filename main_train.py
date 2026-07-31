import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from data.dataset import (
    DentalDataset,
    resolve_cbct_roi_z_bounds,
    resolve_cbct_scale_mode,
)
from losses.registration_loss import (
    registration_loss,
    rotation_error_degrees,
    transform_points,
    translation_error,
)
from models.main_model import RegistrationModel
from utils.domain_correction import affine_signature
from utils.transform_utils import recover_original_transform_torch


def resolve_jaw_config_value(config, key, jaw_type, default):
    """Resolve a shared scalar or an upper/lower value from one YAML file."""
    value = config.get(key, default)
    if isinstance(value, dict):
        if jaw_type not in value:
            raise ValueError(f"{key}.{jaw_type} is required")
        return value[jaw_type]
    return value


@torch.no_grad()
def update_ema_model(ema_model, source_model, decay):
    """Update a validation-only exponential moving average in place."""
    ema_state = ema_model.state_dict()
    source_state = source_model.state_dict()
    for name, ema_value in ema_state.items():
        source_value = source_state[name].detach()
        if ema_value.is_floating_point():
            ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(source_value)


def resolve_pretrained_checkpoint(config, jaw_type):
    checkpoint_map = config.get("pretrained_checkpoints")
    if isinstance(checkpoint_map, dict):
        checkpoint_path = checkpoint_map.get(jaw_type)
        if checkpoint_path:
            return checkpoint_path
    return config.get("pretrained_checkpoint")


def resolve_registration_checkpoint(config, jaw_type):
    checkpoint_map = config.get("registration_initialization_checkpoints")
    if isinstance(checkpoint_map, dict):
        checkpoint_path = checkpoint_map.get(jaw_type)
        if checkpoint_path:
            return checkpoint_path
    return config.get("registration_initialization_checkpoint")


def build_registration_model(config):
    return RegistrationModel(
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
                config,
                'translation_residual_range',
                config['jaw_type'],
                0.0,
            )
        ),
        translation_residual_detach_features=bool(
            resolve_jaw_config_value(
                config,
                'translation_residual_detach_features',
                config['jaw_type'],
                True,
            )
        ),
        translation_residual_hidden_dim=int(
            config.get('translation_residual_hidden_dim', 32)
        ),
    )


def build_data_loader(dataset, config, shuffle):
    num_workers = int(config.get('num_workers', 4))
    generator = torch.Generator()
    generator.manual_seed(int(config.get('training_seed', 2026)))
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': int(config['batch_size']),
        'shuffle': bool(shuffle),
        'num_workers': num_workers,
        'pin_memory': bool(config.get('pin_memory', True)),
        'persistent_workers': bool(
            config.get('persistent_workers', True) and num_workers > 0
        ),
        'generator': generator,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 4))
    return DataLoader(**loader_kwargs)


class AugmentationConcatDataset(ConcatDataset):
    """Concat jaw datasets while preserving augmentation curriculum control."""

    def set_augmentation_magnitude(self, translation, rotation):
        for dataset in self.datasets:
            dataset.set_augmentation_magnitude(translation, rotation)


def move_training_batch(data, device):
    non_blocking = device.type == 'cuda'
    return (
        data['p_src'].to(device, non_blocking=non_blocking),
        data['p_tgt'].to(device, non_blocking=non_blocking),
        data['transform_gt'].to(device, non_blocking=non_blocking),
        data['stl_center'].to(device, non_blocking=non_blocking),
        data['cbct_center'].to(device, non_blocking=non_blocking),
        data['scale'].to(device, non_blocking=non_blocking),
    )


def split_case_ids(dataset, val_fraction, seed, mode="random"):
    """Split cases randomly or hold out complete CBCT affine domains."""
    case_ids = sorted(case_folder.name for case_folder in dataset.case_folders)
    rng = np.random.default_rng(int(seed))
    target_count = max(1, int(round(len(case_ids) * float(val_fraction))))
    target_count = min(target_count, len(case_ids) - 1)

    if mode == "chirality_stratified":
        parity_groups = {"positive": [], "negative": []}
        for case_folder in dataset.case_folders:
            gt_path = (
                dataset.label_dir
                / case_folder.name
                / f"{dataset.jaw_type}_gt.npy"
            )
            determinant = float(np.linalg.det(np.load(gt_path)[:3, :3]))
            parity_groups[
                "negative" if determinant < 0.0 else "positive"
            ].append(case_folder.name)

        if all(parity_groups.values()) and target_count >= 2:
            for values in parity_groups.values():
                rng.shuffle(values)
            negative_count = int(round(
                target_count
                * len(parity_groups["negative"])
                / len(case_ids)
            ))
            negative_count = min(
                max(1, negative_count),
                target_count - 1,
                len(parity_groups["negative"]),
            )
            positive_count = min(
                target_count - negative_count,
                len(parity_groups["positive"]),
            )
            while negative_count + positive_count < target_count:
                if negative_count < len(parity_groups["negative"]):
                    negative_count += 1
                elif positive_count < len(parity_groups["positive"]):
                    positive_count += 1
                else:
                    break
            val_ids = sorted(
                parity_groups["negative"][:negative_count]
                + parity_groups["positive"][:positive_count]
            )
            train_ids = sorted(set(case_ids) - set(val_ids))
            if train_ids and val_ids:
                return train_ids, val_ids, [
                    f"det=-1:{negative_count}",
                    f"det=+1:{positive_count}",
                ]

    if mode == "domain_holdout":
        domains = {}
        for case_folder in dataset.case_folders:
            cbct_img = nib.load(str(case_folder / "CBCT.nii.gz"))
            signature = affine_signature(cbct_img.affine)
            domains.setdefault(signature, []).append(case_folder.name)

        if len(domains) >= 2:
            domain_keys = list(domains)
            rng.shuffle(domain_keys)
            max_val_count = max(target_count, int(round(len(case_ids) * 0.4)))
            selected_domains = []
            selected_count = 0
            remaining = list(domain_keys)
            while remaining:
                feasible = [
                    key
                    for key in remaining
                    if selected_count + len(domains[key]) <= max_val_count
                ]
                if not feasible:
                    break
                best_key = min(
                    feasible,
                    key=lambda key: abs(
                        target_count - (selected_count + len(domains[key]))
                    ),
                )
                new_count = selected_count + len(domains[best_key])
                if selected_domains and abs(target_count - selected_count) <= abs(
                    target_count - new_count
                ):
                    break
                selected_domains.append(best_key)
                remaining.remove(best_key)
                selected_count = new_count
                if selected_count >= target_count:
                    break

            if selected_domains:
                val_ids = sorted(
                    case_id
                    for key in selected_domains
                    for case_id in domains[key]
                )
                train_ids = sorted(set(case_ids) - set(val_ids))
                if train_ids and val_ids:
                    return train_ids, val_ids, selected_domains

    shuffled_ids = list(case_ids)
    rng.shuffle(shuffled_ids)
    val_ids = sorted(shuffled_ids[:target_count])
    train_ids = sorted(shuffled_ids[target_count:])
    return train_ids, val_ids, []


def forward_with_auxiliary_loss(
    model,
    p_src,
    p_tgt,
    transform_gt,
    stl_center,
    cbct_center,
    scale,
    loss_weights,
    auxiliary_pose_weight,
    use_amp=False,
    correspondence_loss_weight=0.0,
    correspondence_radius_mm=5.0,
    correspondence_coordinate_weight=1.0,
    target_saliency_loss_weight=0.0,
    target_saliency_positive_radius_mm=5.0,
    target_saliency_negative_radius_mm=10.0,
    target_saliency_focal_gamma=2.0,
    translation_residual_loss_weight=0.0,
):
    structured_output = model.registration_head_type in {
        'hybrid_direct',
        'cross_attention_svd',
    }
    use_auxiliary = model.registration_head_type == 'hybrid_direct'
    expected_determinant = torch.sign(
        torch.det(transform_gt[:, :3, :3].float())
    ).detach()
    with torch.autocast(
        device_type=p_src.device.type,
        dtype=torch.bfloat16,
        enabled=bool(use_amp and p_src.device.type == 'cuda'),
    ):
        outputs = model(
            p_src,
            p_tgt,
            return_aux=structured_output,
            expected_determinant=expected_determinant,
            source_center_normalized=(
                stl_center / scale.view(-1, 1).clamp_min(1e-6)
            ),
        )
    if structured_output:
        transform_pred = outputs['transform']
    else:
        transform_pred = outputs

    pred_world = recover_original_transform_torch(
        transform_pred, stl_center, cbct_center, scale
    )
    gt_world = recover_original_transform_torch(
        transform_gt, stl_center, cbct_center, scale
    )
    loss, components = registration_loss(
        p_src,
        transform_pred,
        transform_gt,
        loss_weights=loss_weights,
        return_components=True,
        metric_transform_pred=pred_world,
        metric_transform_gt=gt_world,
    )

    translation_residual_mm = 0.0
    translation_residual_loss_value = 0.0
    base_translation_error_mm = components["translation_loss"]
    if structured_output and 'translation_residual' in outputs:
        with torch.no_grad():
            base_world = recover_original_transform_torch(
                outputs['base_transform'], stl_center, cbct_center, scale
            )
            # Adding a normalized residual changes the submitted world
            # translation by ``scale * residual``. Supervise exactly the
            # correction required by the official matrix translation metric,
            # including the current SVD rotation and STL-center contribution.
            residual_target = (
                gt_world[:, :3, 3].float()
                - base_world[:, :3, 3].float()
            ) / scale.float().view(-1, 1).clamp_min(1e-6)
            translation_residual_mm = float(
                (
                    torch.linalg.norm(
                        outputs['translation_residual'].float(), dim=1
                    )
                    * scale.float()
                ).mean().item()
            )
            base_translation_error_mm = float(
                translation_error(base_world, gt_world).mean().item()
            )
        residual_supervision_loss = F.smooth_l1_loss(
            outputs['translation_residual'].float(),
            residual_target.detach(),
            beta=0.02,
        )
        if translation_residual_loss_weight > 0.0:
            loss = (
                loss
                + float(translation_residual_loss_weight)
                * residual_supervision_loss
            )
        translation_residual_loss_value = float(
            residual_supervision_loss.detach().item()
        )

    auxiliary_loss_value = 0.0
    correspondence_loss_value = 0.0
    correspondence_valid_fraction = 0.0
    target_saliency_loss_value = 0.0
    target_saliency_positive_fraction = 0.0
    if use_auxiliary and auxiliary_pose_weight > 0.0:
        auxiliary_losses = []
        for key in ('correspondence_transform', 'global_transform'):
            auxiliary_transform = outputs[key]
            auxiliary_world = recover_original_transform_torch(
                auxiliary_transform, stl_center, cbct_center, scale
            )
            auxiliary_losses.append(
                registration_loss(
                    p_src,
                    auxiliary_transform,
                    transform_gt,
                    loss_weights=loss_weights,
                    metric_transform_pred=auxiliary_world,
                    metric_transform_gt=gt_world,
                )
            )
        auxiliary_loss = torch.stack(auxiliary_losses).mean()
        loss = loss + auxiliary_pose_weight * auxiliary_loss
        auxiliary_loss_value = float(auxiliary_loss.detach().item())

    needs_correspondence = (
        structured_output
        and 'match_logits' in outputs
        and correspondence_loss_weight > 0.0
    )
    needs_saliency = (
        structured_output
        and 'target_saliency_logits' in outputs
        and target_saliency_loss_weight > 0.0
    )
    if needs_correspondence or needs_saliency:
        with torch.no_grad():
            source_gt = transform_points(
                outputs['source_points'].float(), transform_gt.float()
            )
            target_points = outputs['target_points'].float()
            distances = torch.cdist(source_gt, target_points)

        if needs_correspondence:
            logits = outputs['match_logits'].float()
            with torch.no_grad():
                min_distances = distances.min(dim=2).values
                radius_normalized = (
                    float(correspondence_radius_mm)
                    / scale.float().clamp_min(1e-6)
                ).view(-1, 1)
                valid = min_distances <= radius_normalized
                neighborhood = distances <= radius_normalized.unsqueeze(-1)

            if bool(valid.any()):
                # Point-cloud surfaces have no unique point-to-point label.
                log_probabilities = F.log_softmax(logits, dim=2)
                neighborhood_log_mass = torch.logsumexp(
                    log_probabilities.masked_fill(
                        ~neighborhood, float('-inf')
                    ),
                    dim=2,
                )
                classification = -neighborhood_log_mass[valid].mean()
                probabilities = torch.softmax(logits, dim=2)
                predicted_targets = torch.bmm(
                    probabilities, outputs['target_points'].float()
                )
                coordinate = F.smooth_l1_loss(
                    predicted_targets[valid],
                    source_gt[valid],
                    beta=0.02,
                )
                correspondence_loss = (
                    classification
                    + float(correspondence_coordinate_weight) * coordinate
                )
                loss = (
                    loss
                    + float(correspondence_loss_weight)
                    * correspondence_loss
                )
                correspondence_loss_value = float(
                    correspondence_loss.detach().item()
                )
                correspondence_valid_fraction = float(
                    valid.float().mean().item()
                )

        if needs_saliency:
            saliency_logits = outputs['target_saliency_logits'].float()
            with torch.no_grad():
                target_min_distances = distances.min(dim=1).values
                positive_radius = (
                    float(target_saliency_positive_radius_mm)
                    / scale.float().clamp_min(1e-6)
                ).view(-1, 1)
                negative_radius = (
                    float(target_saliency_negative_radius_mm)
                    / scale.float().clamp_min(1e-6)
                ).view(-1, 1)
                positive = target_min_distances <= positive_radius
                negative = target_min_distances >= negative_radius

            if bool(positive.any()) and bool(negative.any()):
                positive_logits = saliency_logits[positive]
                negative_logits = saliency_logits[negative]
                gamma = float(target_saliency_focal_gamma)
                positive_loss = (
                    F.softplus(-positive_logits)
                    * torch.sigmoid(-positive_logits).pow(gamma)
                ).mean()
                negative_loss = (
                    F.softplus(negative_logits)
                    * torch.sigmoid(negative_logits).pow(gamma)
                ).mean()
                target_saliency_loss = 0.5 * (
                    positive_loss + negative_loss
                )
                loss = (
                    loss
                    + float(target_saliency_loss_weight)
                    * target_saliency_loss
                )
                target_saliency_loss_value = float(
                    target_saliency_loss.detach().item()
                )
                target_saliency_positive_fraction = float(
                    positive.float().mean().item()
                )

    components['auxiliary_pose_loss'] = auxiliary_loss_value
    components['correspondence_loss'] = correspondence_loss_value
    components['correspondence_valid_fraction'] = correspondence_valid_fraction
    components['target_saliency_loss'] = target_saliency_loss_value
    components['translation_residual_mm'] = translation_residual_mm
    components[
        'translation_residual_loss'
    ] = translation_residual_loss_value
    components['base_translation_error_mm'] = base_translation_error_mm
    components[
        'target_saliency_positive_fraction'
    ] = target_saliency_positive_fraction
    return transform_pred, pred_world, gt_world, loss, components


def main():
    parser = argparse.ArgumentParser(description="Train a Task2 registration model.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--jaw", choices=["upper", "lower"], default=None)
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.jaw is not None:
        config['jaw_type'] = args.jaw
    if args.experiment_name is not None:
        config['experiment_name'] = args.experiment_name
    else:
        experiment_names = config.get('experiment_names')
        if isinstance(experiment_names, dict):
            config['experiment_name'] = experiment_names[config['jaw_type']]

    training_seed = int(config.get('training_seed', 2026))
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    exp_dir = Path(config['output_dir']) / config['experiment_name']
    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    result_dir = exp_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)
    log_file = open(exp_dir / "log.txt", "a")

    def print_and_log(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()

    print_and_log(f"--- start experiment: {config['experiment_name']} ---")
    print_and_log(f"training seed: {training_seed}")
    print_and_log(f"configuration file:\n{yaml.dump(config)}")
    cbct_roi_z_bounds = resolve_cbct_roi_z_bounds(config, config['jaw_type'])
    cbct_scale_mode = resolve_cbct_scale_mode(config, config['jaw_type'])
    if cbct_roi_z_bounds is not None:
        print_and_log(
            "anatomical CBCT ROI: "
            f"jaw={config['jaw_type']}, relative_z={list(cbct_roi_z_bounds)}, "
            f"center=full_threshold_cloud, scale={cbct_scale_mode}_target_cloud"
        )

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print_and_log(f"use device: {device}")
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get('allow_tf32', True))
        torch.backends.cudnn.allow_tf32 = bool(config.get('allow_tf32', True))
        torch.backends.cudnn.benchmark = bool(config.get('cudnn_benchmark', True))
        print_and_log(
            f"GPU: {torch.cuda.get_device_name(device)} | "
            f"batch_size={config['batch_size']} | "
            f"BF16={bool(config.get('use_bf16', True))} | "
            f"TF32={bool(config.get('allow_tf32', True))}"
        )
    loss_weights = config.get('loss_weights', {"point": 1.0, "rotation": 0.5, "translation": 0.02})
    val_score_weights = config.get('val_score_weights', {"translation": 1.0, "rotation": 1.0})
    val_translation_weight = float(val_score_weights.get("translation", 1.0))
    val_rotation_weight = float(val_score_weights.get("rotation", 1.0))
    print_and_log(f"loss weights: {loss_weights}")
    print_and_log(
        "validation score weights: "
        f"translation={val_translation_weight}, rotation={val_rotation_weight}"
    )

    model = build_registration_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=float(config.get('weight_decay', 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config['epochs'])),
        eta_min=float(config.get('minimum_learning_rate', 1e-6)),
    )

    pretrained_checkpoint = resolve_pretrained_checkpoint(config, config['jaw_type'])
    if config.get('load_pretrained', False) and pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        if 'src_encoder' in checkpoint:
            model.src_encoder.load_state_dict(checkpoint['src_encoder'], strict=False)
        if 'tgt_encoder' in checkpoint:
            model.tgt_encoder.load_state_dict(checkpoint['tgt_encoder'], strict=False)
        print_and_log(f"loaded pretrained encoders from: {pretrained_checkpoint}")

    registration_checkpoint = resolve_registration_checkpoint(
        config, config['jaw_type']
    )
    if config.get('load_registration_initialization', False) and registration_checkpoint:
        checkpoint_path = Path(registration_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Registration initialization not found: {checkpoint_path}"
            )
        state_dict = torch.load(checkpoint_path, map_location=device)
        incompatible = model.load_state_dict(state_dict, strict=False)
        print_and_log(
            f"loaded registration initialization from: {checkpoint_path}"
        )
        print_and_log(
            "initialization unmatched keys: "
            f"missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )

    use_ema = bool(config.get('use_ema', True))
    ema_decay = float(config.get('ema_decay', 0.995))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    ema_model = None
    ema_updates = 0
    if use_ema:
        ema_model = copy.deepcopy(model).eval()
        ema_model.requires_grad_(False)
        print_and_log(f"EMA validation enabled: decay={ema_decay}")

    dataset_kwargs = {
        "data_root": config['train_data_root'],
        "jaw_type": config['jaw_type'],
        "num_points_stl": config['num_points_stl'],
        "num_points_cbct": config['num_points_cbct'],
        "has_labels": True,
        "cbct_threshold": float(config.get('cbct_threshold', 800)),
        "cbct_surface_only": bool(config.get('cbct_surface_only', False)),
        "augmentation_translation": float(config.get('augmentation_translation', 0.35)),
        "augmentation_rotation": float(config.get('augmentation_rotation', 60.0)),
        "use_point_cache": bool(config.get('use_point_cache', True)),
        "point_cache_dir": config.get('point_cache_dir'),
        "rebuild_point_cache": bool(config.get('rebuild_point_cache', False)),
        "stl_cache_points": int(config.get('stl_cache_points', 32768)),
        "cbct_cache_points": int(config.get('cbct_cache_points', 131072)),
        "cbct_roi_z_bounds": cbct_roi_z_bounds,
        "cbct_roi_min_points": int(
            config.get('cbct_roi_min_points', config['num_points_cbct'])
        ),
        "cbct_scale_mode": cbct_scale_mode,
    }

    # The public validation Images often have no Labels directory.  In that
    # case make a deterministic, patient-level split from Train-Labeled so
    # best_model is selected by the same geometric metrics used by the task.
    val_root = Path(config['val_data_root'])
    val_has_labels = any(
        any(label_root.rglob(f"{config['jaw_type']}_gt.npy"))
        for label_root in (val_root / "Labels", val_root / "labels")
        if label_root.is_dir()
    )
    internal_val_fraction = float(config.get("internal_val_fraction", 0.15))
    use_internal_validation = bool(config.get("use_internal_validation", True)) and not val_has_labels
    train_case_ids = None

    if use_internal_validation and internal_val_fraction > 0.0:
        all_dataset = DentalDataset(**dataset_kwargs, use_augmentation=False)
        if bool(config.get('prepare_point_cache_before_training', True)):
            print_and_log(
                f"preparing point cache under: {all_dataset.point_cache_dir}"
            )
            last_report = [0]

            def report_cache_progress(index, total, case_id):
                report_every = max(1, int(config.get('cache_progress_interval', 10)))
                if index == 1 or index == total or index - last_report[0] >= report_every:
                    print_and_log(
                        f"  cache {index}/{total}: case={case_id}"
                    )
                    last_report[0] = index

            all_dataset.prepare_cache(report_cache_progress)
            dataset_kwargs["rebuild_point_cache"] = False
        if len(all_dataset.case_folders) < 2:
            raise RuntimeError("At least two labeled cases are required for an internal validation split")
        split_mode = str(config.get("internal_val_split_mode", "random"))
        train_case_ids, val_case_ids, held_out_groups = split_case_ids(
            all_dataset,
            internal_val_fraction,
            int(config.get("internal_val_seed", 2026)),
            mode=split_mode,
        )
        train_dataset = DentalDataset(
            **dataset_kwargs,
            use_augmentation=True,
            case_ids=train_case_ids,
        )
        val_dataset = DentalDataset(
            **dataset_kwargs,
            use_augmentation=False,
            case_ids=val_case_ids,
            deterministic_sampling=True,
            sampling_seed=int(config.get("internal_val_seed", 2026)),
        )
        print_and_log(
            f"using internal validation split: mode={split_mode}, "
            f"train={len(train_dataset)}, val={len(val_dataset)}, "
            f"seed={config.get('internal_val_seed', 2026)}"
        )
        if held_out_groups:
            print_and_log(
                f"held-out split groups ({len(held_out_groups)}): "
                + ", ".join(held_out_groups)
            )
    else:
        train_dataset = DentalDataset(
            **dataset_kwargs,
            use_augmentation=True,
        )
        val_dataset = None
        if bool(config.get('prepare_point_cache_before_training', True)):
            print_and_log(
                f"preparing point cache under: {train_dataset.point_cache_dir}"
            )
            train_dataset.prepare_cache()

    use_paired_jaw_auxiliary = bool(
        resolve_jaw_config_value(
            config,
            'use_paired_jaw_auxiliary_training',
            config['jaw_type'],
            False,
        )
    )
    if use_paired_jaw_auxiliary:
        auxiliary_jaw = (
            'upper' if config['jaw_type'] == 'lower' else 'lower'
        )
        auxiliary_kwargs = dict(dataset_kwargs)
        auxiliary_kwargs['jaw_type'] = auxiliary_jaw
        auxiliary_dataset = DentalDataset(
            **auxiliary_kwargs,
            use_augmentation=True,
            case_ids=train_case_ids,
        )
        if bool(config.get('prepare_point_cache_before_training', True)):
            print_and_log(
                "preparing paired-jaw auxiliary cache: "
                f"jaw={auxiliary_jaw}, samples={len(auxiliary_dataset)}"
            )
            auxiliary_dataset.prepare_cache()
        primary_samples = len(train_dataset)
        train_dataset = AugmentationConcatDataset(
            [train_dataset, auxiliary_dataset]
        )
        print_and_log(
            "paired-jaw auxiliary training enabled: "
            f"primary_jaw={config['jaw_type']} ({primary_samples}), "
            f"auxiliary_jaw={auxiliary_jaw} ({len(auxiliary_dataset)}), "
            f"total={len(train_dataset)}; validation remains "
            f"{config['jaw_type']}-only"
        )
    train_loader = build_data_loader(train_dataset, config, shuffle=True)

    if val_dataset is not None:
        val_loader = build_data_loader(val_dataset, config, shuffle=False)
        if use_internal_validation:
            print_and_log("validation labels found in Train-Labeled via internal split")
        else:
            print_and_log(f"validation labels found under: {val_root}")
    elif val_has_labels:
        val_dataset = DentalDataset(
            config['val_data_root'],
            config['jaw_type'],
            num_points_stl=config['num_points_stl'],
            num_points_cbct=config['num_points_cbct'],
            use_augmentation=False,
            has_labels=True,
            cbct_threshold=float(config.get('cbct_threshold', 800)),
            cbct_surface_only=bool(config.get('cbct_surface_only', False)),
            deterministic_sampling=True,
            sampling_seed=int(config.get("internal_val_seed", 2026)),
            use_point_cache=bool(config.get('use_point_cache', True)),
            point_cache_dir=config.get('point_cache_dir'),
            rebuild_point_cache=bool(config.get('rebuild_point_cache', False)),
            stl_cache_points=int(config.get('stl_cache_points', 32768)),
            cbct_cache_points=int(config.get('cbct_cache_points', 131072)),
            cbct_roi_z_bounds=cbct_roi_z_bounds,
            cbct_roi_min_points=int(
                config.get('cbct_roi_min_points', config['num_points_cbct'])
            ),
            cbct_scale_mode=cbct_scale_mode,
        )
        if bool(config.get('prepare_point_cache_before_training', True)):
            print_and_log(
                f"preparing validation point cache under: {val_dataset.point_cache_dir}"
            )
            val_dataset.prepare_cache()
        val_loader = build_data_loader(val_dataset, config, shuffle=False)
        print_and_log(f"validation labels found under: {val_root}")
    else:
        val_loader = None
        print_and_log(f"validation labels not found under: {val_root}, skip supervised validation")

    torch.cuda.empty_cache()
    best_val_score = float('inf')
    best_train_loss = float('inf')
    start_time = time.time()

    for epoch in range(config['epochs']):
        epoch_start_time = time.time()

        warmup_epochs = max(1, int(config.get('augmentation_warmup_epochs', 30)))
        augmentation_progress = min(1.0, float(epoch + 1) / warmup_epochs)
        final_translation = float(config.get('augmentation_translation', 0.35))
        final_rotation = float(config.get('augmentation_rotation', 60.0))
        start_translation = float(config.get('augmentation_start_translation', 0.05))
        start_rotation = float(config.get('augmentation_start_rotation', 10.0))
        current_augmentation_translation = (
            start_translation
            + augmentation_progress * (final_translation - start_translation)
        )
        current_augmentation_rotation = (
            start_rotation
            + augmentation_progress * (final_rotation - start_rotation)
        )
        train_dataset.set_augmentation_magnitude(
            current_augmentation_translation, current_augmentation_rotation
        )

        model.train()
        total_train_loss = 0.0
        total_train_point_loss = 0.0
        total_train_rot_loss = 0.0
        total_train_trans_loss = 0.0
        total_train_aux_loss = 0.0
        total_train_correspondence_loss = 0.0
        total_train_correspondence_fraction = 0.0
        total_train_saliency_loss = 0.0
        total_train_saliency_positive_fraction = 0.0
        total_train_translation_residual_mm = 0.0
        total_train_translation_residual_loss = 0.0
        total_train_base_translation_error_mm = 0.0
        total_train_handedness_correct = 0
        total_train_samples = 0

        for data in train_loader:
            p_src, p_tgt, transform_gt, stl_center, cbct_center, scale = (
                move_training_batch(data, device)
            )

            optimizer.zero_grad(set_to_none=True)
            transform_pred, pred_world, gt_world, loss, components = forward_with_auxiliary_loss(
                model,
                p_src,
                p_tgt,
                transform_gt,
                stl_center,
                cbct_center,
                scale,
                loss_weights,
                float(config.get('auxiliary_pose_weight', 0.3)),
                bool(config.get('use_bf16', True)),
                float(config.get('correspondence_loss_weight', 0.1)),
                float(config.get('correspondence_radius_mm', 5.0)),
                float(config.get('correspondence_coordinate_weight', 1.0)),
                float(config.get('target_saliency_loss_weight', 0.2)),
                float(
                    config.get('target_saliency_positive_radius_mm', 5.0)
                ),
                float(
                    config.get('target_saliency_negative_radius_mm', 10.0)
                ),
                float(config.get('target_saliency_focal_gamma', 2.0)),
                float(
                    resolve_jaw_config_value(
                        config,
                        'translation_residual_loss_weight',
                        config['jaw_type'],
                        1.0,
                    )
                ),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.get('gradient_clip_norm', 5.0))
            )
            optimizer.step()
            if ema_model is not None:
                ema_updates += 1
                # Faster adaptation at the beginning, then converge to the
                # configured long-horizon decay.
                effective_decay = min(
                    ema_decay,
                    (1.0 + ema_updates) / (10.0 + ema_updates),
                )
                update_ema_model(ema_model, model, effective_decay)

            total_train_loss += loss.item()
            total_train_point_loss += components["point_loss"]
            total_train_rot_loss += components["rotation_loss_rad"]
            total_train_trans_loss += components["translation_loss"]
            total_train_aux_loss += components["auxiliary_pose_loss"]
            total_train_correspondence_loss += components["correspondence_loss"]
            total_train_correspondence_fraction += components[
                "correspondence_valid_fraction"
            ]
            total_train_saliency_loss += components[
                "target_saliency_loss"
            ]
            total_train_saliency_positive_fraction += components[
                "target_saliency_positive_fraction"
            ]
            total_train_translation_residual_mm += components[
                "translation_residual_mm"
            ]
            total_train_translation_residual_loss += components[
                "translation_residual_loss"
            ]
            total_train_base_translation_error_mm += components[
                "base_translation_error_mm"
            ]
            predicted_sign = torch.det(transform_pred[:, :3, :3])
            target_sign = torch.det(transform_gt[:, :3, :3])
            total_train_handedness_correct += int(
                ((predicted_sign * target_sign) > 0.0).sum().item()
            )
            total_train_samples += p_src.shape[0]

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_point_loss = total_train_point_loss / len(train_loader)
        avg_train_rot_loss = total_train_rot_loss / len(train_loader)
        avg_train_trans_loss = total_train_trans_loss / len(train_loader)
        avg_train_aux_loss = total_train_aux_loss / len(train_loader)
        avg_train_correspondence_loss = (
            total_train_correspondence_loss / len(train_loader)
        )
        avg_train_correspondence_fraction = (
            total_train_correspondence_fraction / len(train_loader)
        )
        avg_train_saliency_loss = (
            total_train_saliency_loss / len(train_loader)
        )
        avg_train_saliency_positive_fraction = (
            total_train_saliency_positive_fraction / len(train_loader)
        )
        avg_train_translation_residual_mm = (
            total_train_translation_residual_mm / len(train_loader)
        )
        avg_train_translation_residual_loss = (
            total_train_translation_residual_loss / len(train_loader)
        )
        avg_train_base_translation_error_mm = (
            total_train_base_translation_error_mm / len(train_loader)
        )
        avg_train_handedness_accuracy = (
            total_train_handedness_correct / max(1, total_train_samples)
        )
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_point', avg_train_point_loss, epoch)
        writer.add_scalar('Loss/train_rotation_rad', avg_train_rot_loss, epoch)
        writer.add_scalar('Loss/train_translation_mm', avg_train_trans_loss, epoch)
        writer.add_scalar('Loss/train_auxiliary_pose', avg_train_aux_loss, epoch)
        writer.add_scalar(
            'Loss/train_correspondence', avg_train_correspondence_loss, epoch
        )
        writer.add_scalar(
            'Correspondence/train_valid_fraction',
            avg_train_correspondence_fraction,
            epoch,
        )
        writer.add_scalar(
            'Loss/train_target_saliency',
            avg_train_saliency_loss,
            epoch,
        )
        writer.add_scalar(
            'TargetSaliency/train_positive_fraction',
            avg_train_saliency_positive_fraction,
            epoch,
        )
        writer.add_scalar(
            'TranslationResidual/train_magnitude_mm',
            avg_train_translation_residual_mm,
            epoch,
        )
        writer.add_scalar(
            'TranslationResidual/train_loss',
            avg_train_translation_residual_loss,
            epoch,
        )
        writer.add_scalar(
            'TranslationResidual/train_base_error_mm',
            avg_train_base_translation_error_mm,
            epoch,
        )
        writer.add_scalar(
            'Metric/train_handedness_accuracy',
            avg_train_handedness_accuracy,
            epoch,
        )
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar(
            'Augmentation/translation', current_augmentation_translation, epoch
        )
        writer.add_scalar(
            'Augmentation/rotation_deg', current_augmentation_rotation, epoch
        )

        model.eval()
        validation_model = ema_model if ema_model is not None else model
        validation_model.eval()
        avg_val_loss = None
        avg_val_trans_mm = None
        avg_val_rot_deg = None
        avg_val_score = None
        avg_val_handedness_accuracy = None
        avg_val_saliency_loss = None
        avg_val_translation_residual_mm = None
        avg_val_base_translation_error_mm = None

        validation_interval = max(1, int(config.get('validation_interval', 1)))
        should_validate = val_loader is not None and (
            epoch == 0
            or (epoch + 1) % validation_interval == 0
            or epoch + 1 == int(config['epochs'])
        )
        if should_validate:
            total_val_loss = 0.0
            total_val_point_loss = 0.0
            total_val_rot_loss = 0.0
            total_val_trans_loss = 0.0
            total_val_trans_mm = 0.0
            total_val_rot_deg = 0.0
            total_val_handedness_correct = 0
            total_val_saliency_loss = 0.0
            total_val_translation_residual_mm = 0.0
            total_val_translation_residual_loss = 0.0
            total_val_base_translation_error_mm = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for data in val_loader:
                    p_src, p_tgt, transform_gt, stl_center, cbct_center, scale = (
                        move_training_batch(data, device)
                    )

                    transform_pred, pred_world, gt_world, loss, components = forward_with_auxiliary_loss(
                        validation_model,
                        p_src,
                        p_tgt,
                        transform_gt,
                        stl_center,
                        cbct_center,
                        scale,
                        loss_weights,
                        0.0,
                        bool(config.get('use_bf16', True)),
                        0.0,
                        float(config.get('correspondence_radius_mm', 5.0)),
                        float(
                            config.get('correspondence_coordinate_weight', 1.0)
                        ),
                        float(config.get('target_saliency_loss_weight', 0.2)),
                        float(
                            config.get(
                                'target_saliency_positive_radius_mm', 5.0
                            )
                        ),
                        float(
                            config.get(
                                'target_saliency_negative_radius_mm', 10.0
                            )
                        ),
                        float(config.get('target_saliency_focal_gamma', 2.0)),
                        0.0,
                    )
                    total_val_loss += loss.item()
                    total_val_point_loss += components["point_loss"]
                    total_val_rot_loss += components["rotation_loss_rad"]
                    total_val_trans_loss += components["translation_loss"]
                    total_val_saliency_loss += components[
                        "target_saliency_loss"
                    ]
                    batch_samples = p_src.shape[0]
                    total_val_translation_residual_mm += (
                        components["translation_residual_mm"]
                        * batch_samples
                    )
                    total_val_translation_residual_loss += (
                        components["translation_residual_loss"]
                        * batch_samples
                    )
                    total_val_base_translation_error_mm += (
                        components["base_translation_error_mm"]
                        * batch_samples
                    )
                    total_val_trans_mm += translation_error(pred_world, gt_world).sum().item()
                    total_val_rot_deg += rotation_error_degrees(pred_world, gt_world).sum().item()
                    predicted_sign = torch.det(transform_pred[:, :3, :3])
                    target_sign = torch.det(transform_gt[:, :3, :3])
                    total_val_handedness_correct += int(
                        ((predicted_sign * target_sign) > 0.0).sum().item()
                    )
                    total_val_samples += p_src.shape[0]

            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_point_loss = total_val_point_loss / len(val_loader)
            avg_val_rot_loss = total_val_rot_loss / len(val_loader)
            avg_val_trans_loss = total_val_trans_loss / len(val_loader)
            avg_val_saliency_loss = (
                total_val_saliency_loss / len(val_loader)
            )
            avg_val_translation_residual_mm = (
                total_val_translation_residual_mm / max(1, total_val_samples)
            )
            avg_val_translation_residual_loss = (
                total_val_translation_residual_loss / max(1, total_val_samples)
            )
            avg_val_base_translation_error_mm = (
                total_val_base_translation_error_mm / max(1, total_val_samples)
            )
            avg_val_trans_mm = total_val_trans_mm / max(1, total_val_samples)
            avg_val_rot_deg = total_val_rot_deg / max(1, total_val_samples)
            avg_val_handedness_accuracy = (
                total_val_handedness_correct / max(1, total_val_samples)
            )
            avg_val_score = (
                val_translation_weight * avg_val_trans_mm
                + val_rotation_weight * avg_val_rot_deg
            )
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('Loss/validation_point', avg_val_point_loss, epoch)
            writer.add_scalar('Loss/validation_rotation_rad', avg_val_rot_loss, epoch)
            writer.add_scalar('Loss/validation_translation_mm', avg_val_trans_loss, epoch)
            writer.add_scalar(
                'Loss/validation_target_saliency',
                avg_val_saliency_loss,
                epoch,
            )
            writer.add_scalar('Metric/validation_translation_mm', avg_val_trans_mm, epoch)
            writer.add_scalar('Metric/validation_rotation_deg', avg_val_rot_deg, epoch)
            writer.add_scalar(
                'TranslationResidual/validation_magnitude_mm',
                avg_val_translation_residual_mm,
                epoch,
            )
            writer.add_scalar(
                'TranslationResidual/validation_loss',
                avg_val_translation_residual_loss,
                epoch,
            )
            writer.add_scalar(
                'TranslationResidual/validation_base_error_mm',
                avg_val_base_translation_error_mm,
                epoch,
            )
            writer.add_scalar(
                'Metric/validation_handedness_accuracy',
                avg_val_handedness_accuracy,
                epoch,
            )
            writer.add_scalar('Metric/validation_score', avg_val_score, epoch)

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time

        if avg_val_loss is None:
            validation_status = (
                "N/A (no labels)" if val_loader is None else "skipped this epoch"
            )
            print_and_log(
                f"Epoch {epoch + 1}/{config['epochs']} | "
                f"Train Loss: {avg_train_loss:.6f} "
                f"(point={avg_train_point_loss:.6f}, rot={avg_train_rot_loss:.6f} rad, "
                f"trans={avg_train_trans_loss:.6f} mm, "
                f"corr={avg_train_correspondence_loss:.6f}, "
                f"sal={avg_train_saliency_loss:.6f}, "
                f"base_trans={avg_train_base_translation_error_mm:.3f} mm, "
                f"tres={avg_train_translation_residual_mm:.3f} mm, "
                f"tres_loss={avg_train_translation_residual_loss:.4f}, "
                f"valid={avg_train_correspondence_fraction:.4f}, "
                f"hand={avg_train_handedness_accuracy:.4f}) | "
                f"Val Loss: {validation_status} | Time: {epoch_time:.2f}s | Total Time: {total_time:.2f}s"
            )
        else:
            print_and_log(
                f"Epoch {epoch + 1}/{config['epochs']} | "
                f"Train Loss: {avg_train_loss:.6f} "
                f"(point={avg_train_point_loss:.6f}, rot={avg_train_rot_loss:.6f} rad, "
                f"trans={avg_train_trans_loss:.6f} mm, "
                f"corr={avg_train_correspondence_loss:.6f}, "
                f"sal={avg_train_saliency_loss:.6f}, "
                f"base_trans={avg_train_base_translation_error_mm:.3f} mm, "
                f"tres={avg_train_translation_residual_mm:.3f} mm, "
                f"tres_loss={avg_train_translation_residual_loss:.4f}, "
                f"valid={avg_train_correspondence_fraction:.4f}, "
                f"hand={avg_train_handedness_accuracy:.4f}) | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Val Metric: trans={avg_val_trans_mm:.6f} mm, rot={avg_val_rot_deg:.6f} deg, "
                f"base_trans={avg_val_base_translation_error_mm:.3f} mm, "
                f"tres={avg_val_translation_residual_mm:.3f} mm, "
                f"sal={avg_val_saliency_loss:.6f}, "
                f"hand={avg_val_handedness_accuracy:.4f}, score={avg_val_score:.6f} | "
                f"Time: {epoch_time:.2f}s | Total Time: {total_time:.2f}s"
            )

        torch.save(model.state_dict(), checkpoint_dir / "latest_model.pth")

        if avg_val_loss is not None:
            checkpoint_model = (
                ema_model if ema_model is not None else model
            )
            if avg_val_score < best_val_score:
                best_val_score = avg_val_score
                torch.save(
                    checkpoint_model.state_dict(),
                    checkpoint_dir / "best_model.pth",
                )
                print_and_log(
                    "  -> Save best model, "
                    f"validation score: {best_val_score:.6f} "
                    f"(trans={avg_val_trans_mm:.6f} mm, rot={avg_val_rot_deg:.6f} deg, "
                    f"hand={avg_val_handedness_accuracy:.4f})"
                )
        elif val_loader is None and avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print_and_log(f"  -> Save best model, training loss: {best_train_loss:.6f}")

        scheduler.step()

    log_file.close()
    writer.close()


if __name__ == '__main__':
    main()
