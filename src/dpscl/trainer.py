"""Single-seed DP-SCL training and evaluation."""

import os

import numpy as np
import torch

from src.dpscl.model import DPSCLModel, SupConLoss

from .config import make_dp_scl_param_dict
from .data import make_loaders, set_seed
from .metrics import compute_metrics_with_threshold, select_threshold_by_f1
from .modes import DP_SCL_MODE, resolve_backend_mode


def eval_dp_scl(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for seq_feat, y_batch in loader:
            seq_feat = seq_feat.to(device)
            logits = model({"batch_size": seq_feat.shape[0], "seq_feat": seq_feat})
            scores.append(torch.sigmoid(logits).detach().cpu().view(-1))
            labels.append(y_batch.detach().cpu().view(-1))
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def train_dp_scl(x, y, train_idx, val_idx, test_idx, args, ds_config, device, checkpoint_dir):
    set_seed(args.current_seed)
    train_loader, val_loader, test_loader = make_loaders(
        x, y, train_idx, val_idx, test_idx, args.batch_size, args.num_workers
    )
    backend_mode = resolve_backend_mode(DP_SCL_MODE)
    model = DPSCLModel(mode=backend_mode, param_dict=make_dp_scl_param_dict(args, ds_config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    supcon = SupConLoss(temperature=args.temperature).to(device)
    checkpoint_path = os.path.join(checkpoint_dir, f"dp_scl_seed_{args.current_seed}.pt")

    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    best_epoch = 0
    patience_count = 0
    stopped_epoch = args.max_epochs
    epoch_history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_bce_sum = 0.0
        train_supcon_sum = 0.0
        train_sample_count = 0

        for seq_feat, y_batch in train_loader:
            seq_feat = seq_feat.to(device)
            y_batch = y_batch.to(device).view(-1, 1)
            optimizer.zero_grad()
            logits, z1, z2 = model({"batch_size": seq_feat.shape[0], "seq_feat": seq_feat})
            bce_loss = bce(logits, y_batch)
            supcon_loss = supcon(torch.stack([z1, z2], dim=1), y_batch.view(-1))
            loss = bce_loss + args.lambda_con * supcon_loss
            loss.backward()
            optimizer.step()

            batch_count = int(y_batch.size(0))
            train_loss_sum += float(loss.detach().cpu()) * batch_count
            train_bce_sum += float(bce_loss.detach().cpu()) * batch_count
            train_supcon_sum += float(supcon_loss.detach().cpu()) * batch_count
            train_sample_count += batch_count

        val_y, val_score = eval_dp_scl(model, val_loader, device)
        val_threshold = select_threshold_by_f1(val_y, val_score)
        val_metrics = compute_metrics_with_threshold(val_y, val_score, val_threshold)
        val_auc = val_metrics["auc"]
        val_f1 = val_metrics["f1"]

        improved = val_auc > best_val_auc + 1e-6 or (
            abs(val_auc - best_val_auc) <= 1e-6 and val_f1 > best_val_f1
        )
        if improved:
            best_val_auc = val_auc
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_sample_count, 1),
            "train_bce_loss": train_bce_sum / max(train_sample_count, 1),
            "train_supcon_loss": train_supcon_sum / max(train_sample_count, 1),
            "val_threshold": val_threshold,
            "val_auc": val_metrics["auc"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "best_val_auc_so_far": best_val_auc,
            "best_val_f1_so_far": best_val_f1,
            "best_epoch_so_far": best_epoch,
            "patience_count": patience_count,
            "is_best": int(best_epoch == epoch),
        })

        if patience_count >= args.patience:
            stopped_epoch = epoch
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    val_y, val_score = eval_dp_scl(model, val_loader, device)
    threshold = select_threshold_by_f1(val_y, val_score)
    test_y, test_score = eval_dp_scl(model, test_loader, device)
    test_metrics = compute_metrics_with_threshold(test_y, test_score, threshold)
    return {
        **test_metrics,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_auc": best_val_auc,
        "epoch_history": epoch_history,
        "status": "ok",
    }
