import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import DentalDataset
from losses.contrastive_loss import nt_xent_loss
from models.pretrain_model import PretrainRegistrationModel


def flatten_points(points):
    batch_size, num_points, _ = points.shape
    points_flat = points.view(-1, 3)
    batch_index = torch.arange(batch_size, device=points.device).repeat_interleave(num_points)
    return points_flat, batch_index


def main():
    config_path = "configs/pretrain_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_dir = Path(config["output_dir"]) / config["experiment_name"]
    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)
    log_file = open(exp_dir / "log.txt", "a", encoding="utf-8")

    def print_and_log(message):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = PretrainRegistrationModel(
        feat_dim=config["feature_dim"],
        proj_dim=config["projection_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    dataset = DentalDataset(
        config["train_data_root"],
        config["jaw_type"],
        num_points_stl=config["num_points_stl"],
        num_points_cbct=config["num_points_cbct"],
        use_augmentation=False,
        has_labels=False,
        mode="pretrain",
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    best_loss = float("inf")
    start_time = time.time()
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for batch in loader:
            src_view1 = batch["p_src_view1"].to(device)
            src_view2 = batch["p_src_view2"].to(device)
            tgt_view1 = batch["p_tgt_view1"].to(device)
            tgt_view2 = batch["p_tgt_view2"].to(device)

            src_view1_flat, src_view1_batch = flatten_points(src_view1)
            src_view2_flat, src_view2_batch = flatten_points(src_view2)
            tgt_view1_flat, tgt_view1_batch = flatten_points(tgt_view1)
            tgt_view2_flat, tgt_view2_batch = flatten_points(tgt_view2)

            optimizer.zero_grad()
            z_src_1 = model.encode_src(src_view1_flat, src_view1_batch)
            z_src_2 = model.encode_src(src_view2_flat, src_view2_batch)
            z_tgt_1 = model.encode_tgt(tgt_view1_flat, tgt_view1_batch)
            z_tgt_2 = model.encode_tgt(tgt_view2_flat, tgt_view2_batch)

            loss_src = nt_xent_loss(z_src_1, z_src_2, temperature=config["temperature"])
            loss_tgt = nt_xent_loss(z_tgt_1, z_tgt_2, temperature=config["temperature"])
            loss_cross = nt_xent_loss(z_src_1, z_tgt_1, temperature=config["temperature"])
            loss = loss_src + loss_tgt + config["cross_modal_weight"] * loss_cross
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        writer.add_scalar("Loss/pretrain", avg_loss, epoch)

        elapsed_epoch = time.time() - epoch_start
        elapsed_total = time.time() - start_time
        print_and_log(
            f"Epoch {epoch + 1}/{config['epochs']} | Loss: {avg_loss:.6f} | "
            f"Time: {elapsed_epoch:.2f}s | Total Time: {elapsed_total:.2f}s"
        )

        checkpoint = {
            "src_encoder": model.src_encoder.state_dict(),
            "tgt_encoder": model.tgt_encoder.state_dict(),
            "src_projection": model.src_projection.state_dict(),
            "tgt_projection": model.tgt_projection.state_dict(),
            "config": config,
            "epoch": epoch + 1,
        }
        torch.save(checkpoint, checkpoint_dir / "latest_model.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print_and_log(f"  -> Save best pretrain model, loss: {best_loss:.6f}")

    log_file.close()
    writer.close()


if __name__ == "__main__":
    main()
