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


def debug_progress_bar(steps_per_epoch: int):
    import random
    import time

    with tqdm.tqdm(total=steps_per_epoch, leave=True, desc="Testing") as pbar:
        for i in range(steps_per_epoch):
            # Sleep for .5 second
            time.sleep(0.005)
            pbar.set_postfix({"loss": random.random()})
            pbar.update(1)

        pbar.set_postfix(
            {
                "ap_off": random.random(),
                "ap_onl": random.random(),
                "rocauc_off": random.random(),
                "rocauc_onl": random.random(),
                "loss": random.random(),
            }
        )


def test_no_log(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    test_steps_per_epoch: int,
    clip_len: int,
    sampling_rate: int,
):
    with torch.no_grad():
        model.eval()
        pred_hlc_full, pred_hl_full, frame_gts_full = [], [], []
        ap_hl, rocauc_hl, ap_hlc, rocauc_hlc = 0.0, 0.0, 0.0, 0.0

        with tqdm.tqdm(total=test_steps_per_epoch, leave=True, desc="Testing") as pbar:
            for i, batch in enumerate(test_loader):
                inputs = batch["feature"].to(device)  # (5, T, D) where 5 is the crop dim
                labels = batch["binary_target"].repeat(5).to(device)  # (1,) -> (5,)
                frame_gts = batch["frame_gts"].numpy()  # (T * clip_len * sampling_rate,)

                logits_hl, logits_hlc = model(inputs, seq_len=None)  # (5, T, 1)
                pred_hlc = torch.mean(torch.sigmoid(torch.squeeze(logits_hlc)), dim=0).detach().cpu().numpy()  # Online detection (T,)
                pred_hl = torch.mean(torch.sigmoid(torch.squeeze(logits_hl)), dim=0).detach().cpu().numpy()  # Offline detection (T,)

                pred_hlc_full.append(pred_hlc)
                pred_hl_full.append(pred_hl)
                frame_gts_full.append(frame_gts)

                # Progress bar
                pbar.update(1)

                # Memory cleanup
                del inputs, labels, frame_gts, logits_hl, logits_hlc
                gc.collect()

            # Concatenate all batches, expand clip preds to frame preds
            pred_hlc_full = np.concatenate(pred_hlc_full).repeat(sampling_rate * clip_len)  # (N * T,) -> (N * T * clip_len * sampling_rate,)
            pred_hl_full = np.concatenate(pred_hl_full).repeat(sampling_rate * clip_len)
            frame_gts_full = np.concatenate(frame_gts_full, dtype=np.int32)  # (N * T * clip_len * sampling_rate,), bool -> int32

            # Compute metrics and losses for an epoch
            ap_hl, _, _ = frame_level_prc_auc(frame_gts_full, pred_hl_full, num_thresholds=10000)
            rocauc_hl, _, _ = frame_level_roc_auc(frame_gts_full, pred_hl_full, num_thresholds=10000)
            ap_hlc, _, _ = frame_level_prc_auc(frame_gts_full, pred_hlc_full, num_thresholds=10000)
            rocauc_hlc, _, _ = frame_level_roc_auc(frame_gts_full, pred_hlc_full, num_thresholds=10000)

            # Log metrics and losses for an epoch
            pbar.set_postfix(
                {
                    "ap_off": ap_hl,
                    "ap_onl": ap_hlc,
                    "rocauc_off": rocauc_hl,
                    "rocauc_onl": rocauc_hlc,
                }
            )

        return {
            "ap_offline": ap_hl,
            "ap_online": ap_hlc,
            "rocauc_offline": rocauc_hl,
            "rocauc_online": rocauc_hlc,
        }


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
):
    with torch.no_grad():
        model.eval()
        pred_hlc_full, pred_hl_full, frame_gts_full = [], [], []
        loss_epoch, distill_epoch, mil_hl_epoch, mil_hlc_epoch, ap_hl, rocauc_hl, ap_hlc, rocauc_hlc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        with tqdm.tqdm(total=test_steps_per_epoch, leave=True, desc="Testing") as pbar:
            for i, batch in enumerate(test_loader):
                inputs = batch["feature"].to(device)  # (5, T, D) where 5 is the crop dim
                labels = batch["binary_target"].repeat(5).to(device)  # (1,) -> (5,)
                frame_gts = batch["frame_gts"].numpy()  # (T * clip_len * sampling_rate,)

                logits_hl, logits_hlc = model(inputs, seq_len=None)  # (5, T, 1)

                # Compute loss for logging
                loss, distill, mil_hl, mil_hlc = criterion(logits_hl, logits_hlc, labels, seq_len=None)
                pred_hlc = torch.mean(torch.sigmoid(torch.squeeze(logits_hlc)), dim=0).detach().cpu().numpy()  # Online detection (T,)
                pred_hl = torch.mean(torch.sigmoid(torch.squeeze(logits_hl)), dim=0).detach().cpu().numpy()  # Offline detection (T,)

                pred_hlc_full.append(pred_hlc)
                pred_hl_full.append(pred_hl)
                frame_gts_full.append(frame_gts)

                loss = loss.cpu().item()
                distill = distill.cpu().item()
                mil_hl = mil_hl.cpu().item()
                mil_hlc = mil_hlc.cpu().item()

                loss_epoch += loss
                distill_epoch += distill
                mil_hl_epoch += mil_hl
                mil_hlc_epoch += mil_hlc

                # Progress bar
                pbar.set_postfix({"loss": loss})
                pbar.update(1)

                # Memory cleanup
                del inputs, labels, frame_gts, logits_hl, logits_hlc, loss, distill, mil_hl, mil_hlc
                gc.collect()

            # Compute average loss for logging
            loss_epoch /= test_steps_per_epoch
            distill_epoch /= test_steps_per_epoch
            mil_hl_epoch /= test_steps_per_epoch
            mil_hlc_epoch /= test_steps_per_epoch

            # Concatenate all batches, expand clip preds to frame preds
            pred_hlc_full = np.concatenate(pred_hlc_full).repeat(sampling_rate * clip_len)  # (N * T,) -> (N * T * clip_len * sampling_rate,)
            pred_hl_full = np.concatenate(pred_hl_full).repeat(sampling_rate * clip_len)
            frame_gts_full = np.concatenate(frame_gts_full, dtype=np.int32)  # (N * T * clip_len * sampling_rate,), bool -> int32

            # Compute metrics and losses for an epoch
            ap_hl, p_hl, r_hl = frame_level_prc_auc(frame_gts_full, pred_hl_full, num_thresholds=10000)
            rocauc_hl, fpr_hl, tpr_hl = frame_level_roc_auc(frame_gts_full, pred_hl_full, num_thresholds=10000)
            ap_hlc, p_hlc, r_hlc = frame_level_prc_auc(frame_gts_full, pred_hlc_full, num_thresholds=10000)
            rocauc_hlc, fpr_hlc, tpr_hlc = frame_level_roc_auc(frame_gts_full, pred_hlc_full, num_thresholds=10000)

            # Log metrics and losses for an epoch
            pbar.set_postfix(
                {
                    "ap_off": ap_hl,
                    "ap_onl": ap_hlc,
                    "rocauc_off": rocauc_hl,
                    "rocauc_onl": rocauc_hlc,
                    "loss": loss_epoch,
                }
            )

            # Plot ROC and PRC curves
            pred_hl_cat = np.stack((1 - pred_hl_full, pred_hl_full), axis=1)  # (N * T * clip_len * sampling_rate, 2), p(normal), p(anomaly)
            pred_hlc_cat = np.stack((1 - pred_hlc_full, pred_hlc_full), axis=1)  # (N * T * clip_len * sampling_rate, 2)
            roc_hl_table = wandb.Table(
                columns=["fpr", "tpr", "rocauc", "epoch"],
                data=list([fpr, tpr, rocauc_hl, current_epoch + 1] for fpr, tpr in zip(fpr_hl, tpr_hl)),
            )
            roc_hlc_table = wandb.Table(
                columns=["fpr", "tpr", "rocauc", "epoch"],
                data=list([fpr, tpr, rocauc_hlc, current_epoch + 1] for fpr, tpr in zip(fpr_hlc, tpr_hlc)),
            )
            prc_hl_table = wandb.Table(
                columns=["recall", "precision", "ap", "epoch"],
                data=list([r, p, ap_hl, current_epoch + 1] for r, p in zip(r_hl, p_hl)),
            )
            prc_hlc_table = wandb.Table(
                columns=["recall", "precision", "ap", "epoch"],
                data=list([r, p, ap_hlc, current_epoch + 1] for r, p in zip(r_hlc, p_hlc)),
            )

            wandb.log(
                {
                    "test/ap_offline": ap_hl,
                    "test/ap_online": ap_hlc,
                    "test/rocauc_offline": rocauc_hl,
                    "test/rocauc_online": rocauc_hlc,
                    "test/loss": loss_epoch,
                    "test/distill_loss": distill_epoch,
                    "test/mil_loss_hl": mil_hl_epoch,
                    "test/mil_loss_hlc": mil_hlc_epoch,
                    "epoch": current_epoch + 1,  # at this epoch
                    "steps_taken": (current_epoch + 1) * train_steps_per_epoch,  # total training steps taken up to this point
                    "test/roc_per_class_offline": wandb.plot.roc_curve(
                        y_true=frame_gts_full,
                        y_probas=pred_hl_cat,
                        labels=["normal", "anomaly"],
                        title="ROC Curve Per Class (Offline)",
                    ),
                    "test/roc_per_class_online": wandb.plot.roc_curve(
                        y_true=frame_gts_full,
                        y_probas=pred_hlc_cat,
                        labels=["normal", "anomaly"],
                        title="ROC Curve Per Class (Online)",
                    ),
                    "test/prc_per_class_offline": wandb.plot.pr_curve(
                        y_true=frame_gts_full,
                        y_probas=pred_hl_cat,
                        labels=["normal", "anomaly"],
                        title="Precision-Recall Curve Per Class (Offline)",
                    ),
                    "test/prc_per_class_online": wandb.plot.pr_curve(
                        y_true=frame_gts_full,
                        y_probas=pred_hlc_cat,
                        labels=["normal", "anomaly"],
                        title="Precision-Recall Curve Per Class (Online)",
                    ),
                    "test/roc_offline": wandb.plot.line(table=roc_hl_table, x="fpr", y="tpr", title="ROC Curve (Offline)"),
                    "test/roc_online": wandb.plot.line(table=roc_hlc_table, x="fpr", y="tpr", title="ROC Curve (Online)"),
                    "test/prc_offline": wandb.plot.line(table=prc_hl_table, x="recall", y="precision", title="Precision-Recall Curve (Offline)"),
                    "test/prc_online": wandb.plot.line(table=prc_hlc_table, x="recall", y="precision", title="Precision-Recall Curve (Online)"),
                },
            )

        return {
            "loss": loss_epoch,
            "ap_offline": ap_hl,
            "ap_online": ap_hlc,
            "rocauc_offline": rocauc_hl,
            "rocauc_online": rocauc_hlc,
        }
