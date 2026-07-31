from itertools import permutations, product
from pathlib import Path

import numpy as np
import yaml


def affine_signature(affine, decimals=2):
    affine = np.asarray(affine, dtype=np.float32)
    values = affine[:3, :4].reshape(-1)
    return "|".join(f"{value:.{decimals}f}" for value in values)


def make_transform(rotation=None, translation=None):
    transform = np.eye(4, dtype=np.float32)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def apply_transform_to_points(points, transform):
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    points_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    return (points_h @ transform.T)[:, :3]


def load_domain_corrections(config_path):
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_corrections = data.get("corrections", data)
    corrections = {}
    for signature, item in raw_corrections.items():
        side = "left"
        matrix = item
        if isinstance(item, dict):
            side = str(item.get("side", "left")).lower()
            matrix = item.get("matrix")
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape == (3, 3):
            matrix = make_transform(rotation=matrix)
        if matrix.shape != (4, 4):
            raise ValueError(f"Invalid correction matrix shape for {signature}: {matrix.shape}")
        if side not in {"left", "right", "conjugate"}:
            raise ValueError(f"Invalid correction side for {signature}: {side}")
        corrections[str(signature)] = {"matrix": matrix, "side": side}
    return corrections


def save_domain_corrections(config_path, corrections):
    serializable = {
        "corrections": {
            signature: {
                "side": value["side"],
                "matrix": np.asarray(value["matrix"], dtype=np.float32).tolist(),
            }
            for signature, value in corrections.items()
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=True)


def generate_right_angle_rotations():
    rotations = []
    seen = set()
    for perm in permutations(range(3)):
        for signs in product([-1.0, 1.0], repeat=3):
            rotation = np.zeros((3, 3), dtype=np.float32)
            for row, col in enumerate(perm):
                rotation[row, col] = signs[row]
            det = round(float(np.linalg.det(rotation)))
            if det != 1:
                continue
            key = tuple(rotation.reshape(-1).tolist())
            if key in seen:
                continue
            seen.add(key)
            rotations.append(make_transform(rotation=rotation))
    return rotations


RIGHT_ANGLE_ROTATIONS = generate_right_angle_rotations()
