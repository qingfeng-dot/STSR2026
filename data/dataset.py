import hashlib
import os

import numpy as np
import nibabel as nib
import torch
import trimesh
import warnings
from pathlib import Path
from scipy.ndimage import binary_erosion
from torch.utils.data import Dataset

from .transforms import PointCloudAugmentor, RandomRigidTransform


def compute_aabb(points):
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    center = (min_corner + max_corner) / 2.0
    return min_corner, max_corner, center


def center_align_points(points, center):
    return points - center


def resolve_cbct_roi_z_bounds(config, jaw_type):
    """Resolve the jaw-specific Z interval shared by training and inference."""
    if not bool(config.get("use_anatomical_cbct_roi", False)):
        return None
    bounds_by_jaw = config.get("cbct_roi_relative_z", {})
    if not isinstance(bounds_by_jaw, dict) or jaw_type not in bounds_by_jaw:
        raise ValueError(
            "use_anatomical_cbct_roi=true requires "
            f"cbct_roi_relative_z.{jaw_type}"
        )
    bounds = np.asarray(bounds_by_jaw[jaw_type], dtype=np.float64)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError(
            f"cbct_roi_relative_z.{jaw_type} must contain two finite values"
        )
    lower, upper = float(bounds[0]), float(bounds[1])
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            f"Invalid cbct_roi_relative_z.{jaw_type}: [{lower}, {upper}]"
        )
    return lower, upper


def resolve_cbct_scale_mode(config, jaw_type):
    """Resolve whether normalization uses the full or sampled target cloud."""
    modes_by_jaw = config.get("cbct_scale_mode", "sampled")
    if isinstance(modes_by_jaw, dict):
        if jaw_type not in modes_by_jaw:
            raise ValueError(f"cbct_scale_mode.{jaw_type} is required")
        mode = str(modes_by_jaw[jaw_type]).lower()
    else:
        mode = str(modes_by_jaw).lower()
    if mode not in {"full", "sampled"}:
        raise ValueError(
            f"Invalid CBCT scale mode for {jaw_type}: {mode}; "
            "expected 'full' or 'sampled'"
        )
    return mode


def crop_cbct_points_by_relative_z(
    points,
    bbox_min,
    bbox_max,
    z_bounds,
    min_output_points=3,
):
    """Crop points in the complete CBCT AABB without changing their frame."""
    if z_bounds is None:
        return points, 1.0, False
    points = np.asarray(points)
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)
    z_span = float(bbox_max[2] - bbox_min[2])
    if not np.isfinite(z_span) or z_span <= 1e-6:
        return points, 1.0, False
    relative_z = (points[:, 2] - bbox_min[2]) / z_span
    keep = (relative_z >= z_bounds[0]) & (relative_z <= z_bounds[1])
    kept_count = int(np.count_nonzero(keep))
    kept_fraction = kept_count / max(1, points.shape[0])
    if kept_count < max(3, int(min_output_points)):
        return points, 1.0, False
    return points[keep], kept_fraction, True


def stable_seed(*parts):
    """Build a process-independent seed (Python's hash is randomized)."""
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def sample_surface_points(mesh, num_points, rng=None):
    """Area-weighted mesh sampling with optional deterministic RNG."""
    if rng is None:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = float(face_areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("Cannot sample a mesh with zero/invalid surface area")

    face_indices = rng.choice(len(mesh.faces), size=num_points, p=face_areas / total_area)
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face_indices]]
    sqrt_u = np.sqrt(rng.random(num_points))[:, None]
    v = rng.random(num_points)[:, None]
    return (
        (1.0 - sqrt_u) * triangles[:, 0]
        + sqrt_u * (1.0 - v) * triangles[:, 1]
        + sqrt_u * v * triangles[:, 2]
    )


def normalize_world_transform(transform, stl_center, cbct_center, scale):
    transform = np.asarray(transform, dtype=np.float32).copy()
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    transform[:3, 3] = (rotation @ stl_center + translation - cbct_center) / scale
    return transform


def resolve_child_dir(root, candidates):
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find any of {candidates} under {root}")


def load_surface_mesh(mesh_path):
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if mesh_path.stat().st_size == 0:
        raise ValueError(f"Mesh file is empty: {mesh_path}")

    # Some trimesh versions return a Scene for STL files containing multiple
    # solids.  Requesting a mesh makes trimesh concatenate those solids.
    try:
        mesh = trimesh.load(str(mesh_path), file_type="stl", force="mesh", process=False)
    except Exception as exc:
        raise ValueError(f"Failed to parse STL mesh {mesh_path}: {exc}") from exc

    if isinstance(mesh, trimesh.Scene):
        geometry = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
        if not geometry:
            geometry_types = [type(g).__name__ for g in mesh.geometry.values()]
            raise ValueError(
                f"No triangular mesh geometry found in {mesh_path}; "
                f"scene geometry types: {geometry_types or ['none']}. "
                "The STL is likely empty, malformed, or contains only points."
            )
        mesh = trimesh.util.concatenate(geometry)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type {type(mesh)} for {mesh_path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Empty mesh found: {mesh_path}")
    return mesh


def atomic_savez(path, **arrays):
    """Write an uncompressed NumPy archive atomically for worker safety."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{stable_seed(path, os.getpid())}.tmp"
    )
    try:
        with open(temporary_path, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class DentalDataset(Dataset):
    def __init__(
        self,
        data_root,
        jaw_type,
        num_points_stl=4096,
        num_points_cbct=8192,
        use_augmentation=False,
        has_labels=True,
        mode="supervised",
        case_ids=None,
        deterministic_sampling=False,
        sampling_seed=2026,
        cbct_threshold=800,
        cbct_surface_only=False,
        augmentation_translation=0.1,
        augmentation_rotation=30.0,
        use_point_cache=True,
        point_cache_dir=None,
        rebuild_point_cache=False,
        stl_cache_points=32768,
        cbct_cache_points=131072,
        cbct_roi_z_bounds=None,
        cbct_roi_min_points=None,
        cbct_scale_mode="sampled",
    ):
        assert jaw_type in ["lower", "upper"], "jaw_type must be 'lower' or 'upper'"
        assert mode in ["supervised", "pretrain"], "mode must be 'supervised' or 'pretrain'"

        self.data_root = Path(data_root)
        self.jaw_type = jaw_type
        self.num_points_stl = num_points_stl
        self.num_points_cbct = num_points_cbct
        self.use_augmentation = use_augmentation
        self.has_labels = has_labels
        self.mode = mode
        self.deterministic_sampling = deterministic_sampling
        self.sampling_seed = int(sampling_seed)
        self.cbct_threshold = float(cbct_threshold)
        self.cbct_surface_only = bool(cbct_surface_only)
        self.use_point_cache = bool(use_point_cache)
        self.rebuild_point_cache = bool(rebuild_point_cache)
        self.stl_cache_points = max(int(stl_cache_points), int(num_points_stl))
        self.cbct_cache_points = max(int(cbct_cache_points), int(num_points_cbct))
        self.cbct_roi_z_bounds = (
            tuple(float(value) for value in cbct_roi_z_bounds)
            if cbct_roi_z_bounds is not None
            else None
        )
        self.cbct_roi_min_points = max(
            3,
            int(cbct_roi_min_points)
            if cbct_roi_min_points is not None
            else int(num_points_cbct),
        )
        self.cbct_scale_mode = str(cbct_scale_mode).lower()
        if self.cbct_scale_mode not in {"full", "sampled"}:
            raise ValueError(
                f"Invalid cbct_scale_mode: {self.cbct_scale_mode}"
            )
        self.point_cache_dir = (
            Path(point_cache_dir).expanduser()
            if point_cache_dir
            else self.data_root / ".task2_point_cache"
        )

        # DataLoader persistent workers own dataset copies.  Shared-memory
        # values keep the augmentation curriculum synchronized across epochs.
        self.augmentation_magnitude = torch.tensor(
            [float(augmentation_translation), float(augmentation_rotation)],
            dtype=torch.float64,
        ).share_memory_()

        self.transform_aug = RandomRigidTransform(
            mag_trans=float(augmentation_translation),
            mag_rot=float(augmentation_rotation),
        )
        self.pretrain_aug = PointCloudAugmentor(mag_trans=0.1, mag_rot=30, jitter_std=0.01, dropout_rate=0.1)
        self.image_dir = resolve_child_dir(self.data_root, ["Images", "images"])
        self.label_dir = resolve_child_dir(self.data_root, ["Labels", "labels"]) if self.has_labels else None

        candidate_folders = sorted(
            [p for p in self.image_dir.iterdir() if p.is_dir() and (p / f"{self.jaw_type}.stl").exists()]
        )
        if case_ids is not None:
            allowed_case_ids = {str(case_id) for case_id in case_ids}
            candidate_folders = [p for p in candidate_folders if p.name in allowed_case_ids]
        if self.has_labels and self.label_dir is not None:
            candidate_folders = [
                p
                for p in candidate_folders
                if (self.label_dir / p.name / f"{self.jaw_type}_gt.npy").exists()
            ]
        self.case_folders = []
        for case_folder in candidate_folders:
            stl_path = case_folder / f"{self.jaw_type}.stl"
            try:
                if not self._has_valid_stl_cache(stl_path):
                    load_surface_mesh(stl_path)
            except (OSError, TypeError, ValueError) as exc:
                warnings.warn(
                    f"Skipping invalid case {case_folder.name}: {exc}",
                    RuntimeWarning,
                )
                continue
            self.case_folders.append(case_folder)

        if not self.case_folders:
            raise RuntimeError(
                f"No valid {self.jaw_type} STL cases found under {self.image_dir}"
            )

    def __len__(self):
        return len(self.case_folders)

    def set_augmentation_magnitude(self, translation, rotation):
        self.augmentation_magnitude[0] = float(translation)
        self.augmentation_magnitude[1] = float(rotation)

    def _source_signature(self, path, *settings):
        stat = Path(path).stat()
        return hashlib.sha256(
            "|".join(
                [str(Path(path).resolve()), str(stat.st_size), str(stat.st_mtime_ns)]
                + [str(value) for value in settings]
            ).encode("utf-8")
        ).hexdigest()[:16]

    def _stl_cache_path(self, stl_path):
        signature = self._source_signature(stl_path, self.stl_cache_points, "v1")
        return (
            self.point_cache_dir
            / "stl"
            / f"{Path(stl_path).parent.name}_{self.jaw_type}_{signature}.npz"
        )

    def _cbct_cache_path(self, cbct_path):
        signature = self._source_signature(
            cbct_path,
            self.cbct_threshold,
            int(self.cbct_surface_only),
            self.cbct_cache_points,
            "v2",
        )
        return (
            self.point_cache_dir
            / "cbct"
            / f"{Path(cbct_path).parent.name}_{signature}.npz"
        )

    def _has_valid_stl_cache(self, stl_path):
        if not self.use_point_cache or self.rebuild_point_cache:
            return False
        cache_path = self._stl_cache_path(stl_path)
        if not cache_path.is_file():
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                return cached["points"].shape == (self.stl_cache_points, 3)
        except (OSError, ValueError, KeyError):
            return False

    def _load_stl_data(self, stl_path):
        cache_path = self._stl_cache_path(stl_path)
        if self.use_point_cache and cache_path.is_file() and not self.rebuild_point_cache:
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    return (
                        cached["points"].astype(np.float32, copy=False),
                        cached["bbox_min"].astype(np.float32, copy=False),
                        cached["bbox_max"].astype(np.float32, copy=False),
                        cached["center"].astype(np.float32, copy=False),
                    )
            except (OSError, ValueError, KeyError):
                warnings.warn(f"Rebuilding invalid STL cache: {cache_path}")

        mesh = load_surface_mesh(stl_path)
        bbox_min, bbox_max, center = compute_aabb(mesh.vertices)
        cache_rng = np.random.default_rng(
            stable_seed("stl-cache", Path(stl_path).resolve(), self.stl_cache_points)
        )
        points = sample_surface_points(
            mesh, self.stl_cache_points, rng=cache_rng
        ).astype(np.float32)
        if self.use_point_cache:
            atomic_savez(
                cache_path,
                points=points,
                bbox_min=np.asarray(bbox_min, dtype=np.float32),
                bbox_max=np.asarray(bbox_max, dtype=np.float32),
                center=np.asarray(center, dtype=np.float32),
            )
        return (
            points,
            np.asarray(bbox_min, dtype=np.float32),
            np.asarray(bbox_max, dtype=np.float32),
            np.asarray(center, dtype=np.float32),
        )

    def _load_cbct_data(self, cbct_path):
        cache_path = self._cbct_cache_path(cbct_path)
        if self.use_point_cache and cache_path.is_file() and not self.rebuild_point_cache:
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    return (
                        cached["points"].astype(np.float32, copy=False),
                        cached["bbox_min"].astype(np.float32, copy=False),
                        cached["bbox_max"].astype(np.float32, copy=False),
                        cached["center"].astype(np.float32, copy=False),
                    )
            except (OSError, ValueError, KeyError):
                warnings.warn(f"Rebuilding invalid CBCT cache: {cache_path}")

        points = self._sample_cbct_points(cbct_path).astype(np.float32)
        bbox_min, bbox_max, center = compute_aabb(points)
        if points.shape[0] > self.cbct_cache_points:
            cache_rng = np.random.default_rng(
                stable_seed(
                    "cbct-cache",
                    Path(cbct_path).resolve(),
                    self.cbct_threshold,
                    self.cbct_surface_only,
                    self.cbct_cache_points,
                )
            )
            indices = cache_rng.choice(
                points.shape[0], self.cbct_cache_points, replace=False
            )
            points = points[indices]
        if self.use_point_cache:
            atomic_savez(
                cache_path,
                points=points,
                bbox_min=np.asarray(bbox_min, dtype=np.float32),
                bbox_max=np.asarray(bbox_max, dtype=np.float32),
                center=np.asarray(center, dtype=np.float32),
            )
        return (
            points,
            np.asarray(bbox_min, dtype=np.float32),
            np.asarray(bbox_max, dtype=np.float32),
            np.asarray(center, dtype=np.float32),
        )

    def prepare_cache(self, progress_callback=None):
        """Build reusable STL and CBCT point archives before training."""
        if not self.use_point_cache:
            return
        total = len(self.case_folders)
        for index, case_folder in enumerate(self.case_folders, start=1):
            self._load_stl_data(case_folder / f"{self.jaw_type}.stl")
            self._load_cbct_data(case_folder / "CBCT.nii.gz")
            if progress_callback is not None:
                progress_callback(index, total, case_folder.name)
        # ``rebuild`` means rebuild once at startup, not on every __getitem__.
        self.rebuild_point_cache = False

    def _sample_cbct_points(self, cbct_path):
        cbct_img = nib.load(str(cbct_path))
        cbct_data = cbct_img.get_fdata()

        mask = cbct_data > self.cbct_threshold
        if self.cbct_surface_only:
            eroded = binary_erosion(
                mask,
                structure=np.ones((3, 3, 3), dtype=bool),
                border_value=0,
            )
            surface_mask = mask & ~eroded
            if np.any(surface_mask):
                mask = surface_mask

        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            raise ValueError(
                f"No CBCT points found above threshold {self.cbct_threshold} in {cbct_path}"
            )
        coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
        points = (cbct_img.affine @ coords_h.T).T[:, :3]
        return points

    def _resample_points(self, points, num_points, rng=None):
        replace = points.shape[0] < num_points
        chooser = rng if rng is not None else np.random
        indices = chooser.choice(points.shape[0], num_points, replace=replace)
        return points[indices]

    def _load_case_points(self, case_folder):
        patient_id = case_folder.name
        rng = None
        if self.deterministic_sampling:
            rng = np.random.default_rng(stable_seed(self.sampling_seed, patient_id, self.jaw_type))
        stl_path = case_folder / f"{self.jaw_type}.stl"
        cbct_path = case_folder / "CBCT.nii.gz"

        stl_pool, stl_min, stl_max, stl_center = self._load_stl_data(stl_path)
        p_src = self._resample_points(stl_pool, self.num_points_stl, rng=rng)
        p_src = center_align_points(p_src, stl_center)

        p_tgt, cbct_min, cbct_max, cbct_center = self._load_cbct_data(cbct_path)
        full_centered_target = center_align_points(p_tgt, cbct_center)
        full_scale = np.max(np.linalg.norm(full_centered_target, axis=1))
        p_tgt, roi_kept_fraction, roi_applied = crop_cbct_points_by_relative_z(
            p_tgt,
            cbct_min,
            cbct_max,
            self.cbct_roi_z_bounds,
            min_output_points=self.cbct_roi_min_points,
        )
        if self.cbct_roi_z_bounds is not None and not roi_applied:
            warnings.warn(
                f"Skipping anatomical CBCT ROI for case {patient_id}: "
                f"fewer than {self.cbct_roi_min_points} points would remain",
                RuntimeWarning,
            )
        p_tgt = center_align_points(p_tgt, cbct_center)
        p_tgt = self._resample_points(p_tgt, self.num_points_cbct, rng=rng)

        if self.cbct_scale_mode == "full":
            scale = full_scale
        else:
            # Reproduce the original baseline/pretraining distribution: the
            # scale is derived from exactly the target points fed to the net.
            scale = np.max(np.linalg.norm(p_tgt, axis=1))
        if scale < 1e-6:
            scale = 1.0

        p_src = p_src / scale
        p_tgt = p_tgt / scale

        meta = {
            "patient_id": patient_id,
            "stl_center": torch.from_numpy(stl_center).float(),
            "cbct_center": torch.from_numpy(cbct_center).float(),
            "stl_bbox_min": torch.from_numpy(stl_min).float(),
            "stl_bbox_max": torch.from_numpy(stl_max).float(),
            "cbct_bbox_min": torch.from_numpy(cbct_min).float(),
            "cbct_bbox_max": torch.from_numpy(cbct_max).float(),
            "cbct_roi_kept_fraction": torch.tensor(roi_kept_fraction).float(),
            "cbct_roi_applied": torch.tensor(roi_applied, dtype=torch.bool),
            "scale": torch.tensor(scale).float(),
        }
        return p_src.astype(np.float32), p_tgt.astype(np.float32), meta

    def _build_pretrain_item(self, p_src, p_tgt, meta):
        src_view1 = self.pretrain_aug(p_src, target_num_points=self.num_points_stl).astype(np.float32)
        src_view2 = self.pretrain_aug(p_src, target_num_points=self.num_points_stl).astype(np.float32)
        tgt_view1 = self.pretrain_aug(p_tgt, target_num_points=self.num_points_cbct).astype(np.float32)
        tgt_view2 = self.pretrain_aug(p_tgt, target_num_points=self.num_points_cbct).astype(np.float32)

        return {
            "p_src_view1": torch.from_numpy(src_view1).float(),
            "p_src_view2": torch.from_numpy(src_view2).float(),
            "p_tgt_view1": torch.from_numpy(tgt_view1).float(),
            "p_tgt_view2": torch.from_numpy(tgt_view2).float(),
            **meta,
        }

    def __getitem__(self, idx):
        case_folder = self.case_folders[idx]
        patient_id = case_folder.name

        p_src, p_tgt, meta = self._load_case_points(case_folder)
        if self.mode == "pretrain":
            return self._build_pretrain_item(p_src, p_tgt, meta)

        transform_gt = np.eye(4, dtype=np.float32)
        has_gt = False
        if self.has_labels and self.label_dir is not None:
            gt_path = self.label_dir / patient_id / f"{self.jaw_type}_gt.npy"
            if gt_path.exists():
                transform_gt = np.load(str(gt_path)).astype(np.float32)
                has_gt = True

        if has_gt:
            transform_gt = normalize_world_transform(
                transform_gt,
                meta["stl_center"].numpy(),
                meta["cbct_center"].numpy(),
                meta["scale"].item(),
            )

        if self.use_augmentation:
            self.transform_aug.mag_trans = float(self.augmentation_magnitude[0].item())
            self.transform_aug.mag_rot = float(self.augmentation_magnitude[1].item())
            p_src_aug, transform_aug = self.transform_aug(p_src)
            if has_gt:
                transform_gt = transform_gt @ np.linalg.inv(transform_aug)
            p_src = p_src_aug.astype(np.float32)

        return {
            "p_src": torch.from_numpy(p_src).float(),
            "p_tgt": torch.from_numpy(p_tgt).float(),
            "transform_gt": torch.from_numpy(transform_gt).float(),
            "has_gt": has_gt,
            **meta,
        }
