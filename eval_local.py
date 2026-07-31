from pathlib import Path
import argparse

import numpy as np


def rotation_error_deg(pred, gt):
    r_pred = pred[:3, :3]
    r_gt = gt[:3, :3]
    r_rel = r_pred @ r_gt.T
    cos_theta = (np.trace(r_rel) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error_mm(pred, gt):
    t_pred = pred[:3, 3]
    t_gt = gt[:3, 3]
    return float(np.linalg.norm(t_pred - t_gt))


def evaluate_case(pred_path, gt_path):
    pred = np.load(str(pred_path)).astype(np.float64)
    gt = np.load(str(gt_path)).astype(np.float64)
    return translation_error_mm(pred, gt), rotation_error_deg(pred, gt)


def main():
    parser = argparse.ArgumentParser(description="Evaluate local Task2 predictions against labeled ground truth.")
    parser.add_argument("--prediction_root", type=str, required=True, help="Folder containing case subfolders with upper_gt.npy/lower_gt.npy")
    parser.add_argument("--label_root", type=str, required=True, help="Folder containing ground-truth case subfolders")
    parser.add_argument("--jaw", type=str, choices=["upper", "lower", "both"], default="both")
    args = parser.parse_args()

    prediction_root = Path(args.prediction_root)
    label_root = Path(args.label_root)

    if not prediction_root.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {prediction_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Label root not found: {label_root}")

    jaws = ["upper", "lower"] if args.jaw == "both" else [args.jaw]
    records = []
    missing = []

    case_ids = sorted([p.name for p in label_root.iterdir() if p.is_dir()])
    for case_id in case_ids:
        for jaw in jaws:
            pred_path = prediction_root / case_id / f"{jaw}_gt.npy"
            gt_path = label_root / case_id / f"{jaw}_gt.npy"
            if not gt_path.exists():
                continue
            if not pred_path.exists():
                missing.append(str(pred_path))
                continue
            trans_err, rot_err = evaluate_case(pred_path, gt_path)
            records.append((case_id, jaw, trans_err, rot_err))

    if not records:
        raise RuntimeError("No valid prediction/ground-truth pairs were found.")

    mean_trans = float(np.mean([r[2] for r in records]))
    mean_rot = float(np.mean([r[3] for r in records]))

    print(f"Pairs evaluated: {len(records)}")
    print(f"Mean_Translation_Error_mm: {mean_trans:.6f}")
    print(f"Mean_Rotation_Error_deg: {mean_rot:.6f}")

    worst_by_trans = sorted(records, key=lambda x: x[2], reverse=True)[:10]
    worst_by_rot = sorted(records, key=lambda x: x[3], reverse=True)[:10]

    print("\nTop 10 worst by translation:")
    for case_id, jaw, trans_err, rot_err in worst_by_trans:
        print(f"{case_id} {jaw} | trans={trans_err:.6f} mm | rot={rot_err:.6f} deg")

    print("\nTop 10 worst by rotation:")
    for case_id, jaw, trans_err, rot_err in worst_by_rot:
        print(f"{case_id} {jaw} | trans={trans_err:.6f} mm | rot={rot_err:.6f} deg")

    if missing:
        print(f"\nMissing predictions: {len(missing)}")
        for path in missing[:20]:
            print(path)
        if len(missing) > 20:
            print("...")


if __name__ == "__main__":
    main()
