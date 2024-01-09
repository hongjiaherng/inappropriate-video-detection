from typing import Dict
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import tqdm.auto as tqdm
import wandb
from metrics import (
    frame_level_prc_auc,
    frame_level_roc_auc,
)


def test_no_log(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    test_steps_per_epoch: int,
    clip_len: int,
    sampling_rate: int,
) -> Dict[str, float]:
    with torch.no_grad():
        model.eval()
        pred_full, frame_gts_full = [], []

        with tqdm.tqdm(total=test_steps_per_epoch, leave=True, desc="Testing") as pbar:
            for i, batch in enumerate(test_loader):
                inputs = batch["feature"].to(device)  # (5, T, D) where 5 is the crop dim
                n_crops, seq_len = inputs.shape[0], inputs.shape[1]
                binary_target = batch["binary_target"].repeat(n_crops * seq_len).view(-1, 1).to(device)  # (,) -> (5 * T * 1)
                frame_gts = batch["frame_gts"].numpy()  # (T * clip_len * sampling_rate,)

                inputs = inputs.view(seq_len * n_crops, -1).to(device)  # (5, T, D) -> (5 * T, D)
                logits = model(inputs)  # (5 * T, 1)

                pred = torch.sigmoid(logits)
                pred = pred.view(n_crops, seq_len, -1)  # (5 * T, 1) -> (5, T, 1)
                pred = torch.mean(pred, dim=0).detach().cpu().numpy()  # (T, 1)

                pred_full.append(pred)
                frame_gts_full.append(frame_gts)

                pbar.update(1)

                # Memory cleanup
                del inputs, binary_target, frame_gts, logits, pred
                gc.collect()

        # Concatenate all batches, expand clip preds to frame preds
        pred_full = np.concatenate(pred_full).repeat(sampling_rate * clip_len)  # (N * T,) -> (N * T * clip_len * sampling_rate,)
        frame_gts_full = np.concatenate(frame_gts_full, dtype=np.int32)  # (N * T * clip_len * sampling_rate,), bool -> int32

        # Compute metrics for an epoch
        ap, p, r = frame_level_prc_auc(frame_gts_full, pred_full, num_thresholds=10000)
        rocauc, fpr, tpr = frame_level_roc_auc(frame_gts_full, pred_full, num_thresholds=10000)

        # Log metrics and losses for an epoch
        pbar.set_postfix({"ap": ap, "rocauc": rocauc})

        return {"ap": ap, "rocauc": rocauc}


def test_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    test_steps_per_epoch: int,
    train_steps_per_epoch: int,  # For logging
    current_epoch: int,
    clip_len: int,
    sampling_rate: int,
) -> Dict[str, float]:
    with torch.no_grad():
        model.eval()
        pred_full, frame_gts_full = [], []
        loss_epoch = 0.0

        with tqdm.tqdm(total=test_steps_per_epoch, leave=True, desc="Testing") as pbar:
            for i, batch in enumerate(test_loader):
                inputs = batch["feature"].to(device)  # (5, T, D) where 5 is the crop dim
                n_crops, seq_len = inputs.shape[0], inputs.shape[1]
                binary_target = batch["binary_target"].repeat(n_crops * seq_len).view(-1, 1).to(device)  # (,) -> (5 * T * 1)
                frame_gts = batch["frame_gts"].numpy()  # (T * clip_len * sampling_rate,)

                inputs = inputs.view(seq_len * n_crops, -1).to(device)  # (5, T, D) -> (5 * T, D)
                logits = model(inputs)  # (5 * T, 1)

                loss = criterion(logits, binary_target)  # (1,)
                pred = torch.sigmoid(logits)
                pred = pred.view(n_crops, seq_len, -1)  # (5 * T, 1) -> (5, T, 1)
                pred = torch.mean(pred, dim=0).detach().cpu().numpy()  # (T, 1)

                pred_full.append(pred)
                frame_gts_full.append(frame_gts)

                loss = loss.detach().cpu().item()
                loss_epoch += loss

                # Progress bar
                pbar.set_postfix({"loss": loss})
                pbar.update(1)

                # Memory cleanup
                del inputs, binary_target, frame_gts, logits, loss, pred
                gc.collect()

            # Compute average loss for an epoch
            loss_epoch = loss_epoch / test_steps_per_epoch

            # Concatenate all batches, expand clip preds to frame preds
            pred_full = np.concatenate(pred_full).repeat(sampling_rate * clip_len)  # (N * T,) -> (N * T * clip_len * sampling_rate,)
            frame_gts_full = np.concatenate(frame_gts_full, dtype=np.int32)  # (N * T * clip_len * sampling_rate,), bool -> int32

            # Compute metrics for an epoch
            ap, p, r = frame_level_prc_auc(frame_gts_full, pred_full, num_thresholds=10000)
            rocauc, fpr, tpr = frame_level_roc_auc(frame_gts_full, pred_full, num_thresholds=10000)

            # Log metrics and losses for an epoch
            pbar.set_postfix({"ap": ap, "rocauc": rocauc, "loss": loss_epoch})

            # Plot ROC and PRC curves
            pred_cat = np.stack((1 - pred_full, pred_full), axis=1)  # (N * T * clip_len * sampling_rate, 2) = p(normal), p(anomaly)
            roc_table = wandb.Table(
                columns=["fpr", "tpr", "rocauc", "epoch"],
                data=list([fpr, tpr, rocauc, current_epoch + 1] for fpr, tpr in zip(fpr, tpr)),
            )
            prc_table = wandb.Table(
                columns=["recall", "precision", "ap", "epoch"],
                data=list([r, p, ap, current_epoch + 1] for r, p in zip(r, p)),
            )

            wandb.log(
                {
                    "test/ap": ap,
                    "test/rocauc": rocauc,
                    "test/loss": loss_epoch,
                    "epoch": current_epoch + 1,  # at this epoch
                    "steps_taken": (current_epoch + 1) * train_steps_per_epoch,  # total training steps taken up to this point
                    "test/roc_per_class": wandb.plot.roc_curve(
                        y_true=frame_gts_full,
                        y_probas=pred_cat,
                        labels=["normal", "anomaly"],
                        title="ROC Curve Per Class",
                    ),
                    "test/prc_per_class": wandb.plot.pr_curve(
                        y_true=frame_gts_full,
                        y_probas=pred_cat,
                        labels=["normal", "anomaly"],
                        title="Precision-Recall Curve Per Class",
                    ),
                    "test/roc": wandb.plot.line(table=roc_table, x="fpr", y="tpr", title="ROC Curve"),
                    "test/prc": wandb.plot.line(table=prc_table, x="recall", y="precision", title="Precision-Recall Curve"),
                },
            )

        return {"ap": ap, "rocauc": rocauc, "loss": loss_epoch}
