from typing import Dict
import gc

import torch
import torch.nn as nn
import torch.utils.data
import tqdm.auto as tqdm
import wandb


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    normal_loader: torch.utils.data.DataLoader,
    anomaly_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps_per_epoch: int,
    current_epoch: int,
    log_interval_steps: int,
) -> Dict[str, float]:
    loss_epoch, sparsity_epoch, smoothness_epoch = 0.0, 0.0, 0.0

    normal_iter = iter(normal_loader)
    anomaly_iter = iter(anomaly_loader)

    model.train()
    with tqdm.tqdm(total=steps_per_epoch, desc="Training", leave=True) as pbar:
        for i in range(steps_per_epoch):
            inputs_normal = next(normal_iter)["feature"]  # (B, T, D)
            inputs_anomaly = next(anomaly_iter)["feature"]  # (B, T, D)

            assert inputs_normal.shape[0] == inputs_anomaly.shape[0], "Batch sizes are not equal"
            assert inputs_normal.shape[1] == inputs_anomaly.shape[1], "Sequence lengths are not equal"

            batch_size = inputs_normal.shape[0]
            seq_len = inputs_normal.shape[1]

            inputs_normal = inputs_normal.view(batch_size * seq_len, -1)  # (B * T, D)
            inputs_anomaly = inputs_anomaly.view(batch_size * seq_len, -1)  # (B * T, D)

            inputs = torch.cat([inputs_normal, inputs_anomaly], dim=0).to(device)  # (2B * T, D)
            scores = model(inputs)  # (2B * T, 1)

            scores_normal = scores[: batch_size * seq_len]  # (B * T, 1)
            scores_anomaly = scores[batch_size * seq_len :]  # (B * T, 1)
            loss, sparsity, smoothness = criterion(scores_normal, scores_anomaly)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For logging
            loss = loss.detach().cpu().item()
            sparsity = sparsity.detach().cpu().item()
            smoothness = smoothness.detach().cpu().item()

            loss_epoch += loss
            sparsity_epoch += sparsity
            smoothness_epoch += smoothness

            # Step-level logging (log every n steps)
            if i == 0 or (i + 1) % log_interval_steps == 0:
                wandb.log(
                    {
                        "train/loss_step": loss,
                        "train/sparsity_step": sparsity,
                        "train/smoothness_step": smoothness,
                        "steps_taken": i + 1 + current_epoch * steps_per_epoch,
                    },
                )

            pbar.set_postfix({"loss": loss, "sparsity": sparsity, "smoothness": smoothness})
            pbar.update(1)

            # Memory cleanup
            del inputs, inputs_normal, inputs_anomaly, scores, scores_normal, scores_anomaly, loss, sparsity, smoothness
            gc.collect()

        # Epoch-level logging (log every epoch)
        loss_epoch /= steps_per_epoch
        sparsity_epoch /= steps_per_epoch
        smoothness_epoch /= steps_per_epoch

        wandb.log(
            {
                "train/loss": loss_epoch,
                "train/sparsity": sparsity_epoch,
                "train/smoothness": smoothness_epoch,
                "epoch": current_epoch + 1,  # at this epoch
                "steps_taken": (current_epoch + 1) * steps_per_epoch,  # total steps taken up to this point
            },
        )
        pbar.set_postfix({"loss": loss_epoch, "sparsity": sparsity_epoch, "smoothness": smoothness_epoch})

    return {
        "loss": loss_epoch,
        "sparsity": sparsity_epoch,
        "smoothness": smoothness_epoch,
    }
