import argparse
from utils import int_or_none, str_or_none, str2bool


CONFIG_SHAPE = {
    "training_cfg": ["max_epochs", "batch_size", "seed", "pretrained_ckpt"],
    "optimizer_cfg": ["lr", {"lr_scheduler": ["milestones", "gamma"]}],
    "model_cfg": ["model_name", "dropout_prob", "hlc_ctx_len", "threshold", "sigma", "gamma", {"loss": ["lambda", "is_topk", "q"]}],
    "dataset_cfg": ["feature_name", "seperated_by_class", "feature_dim", "clip_len", "sampling_rate", "streaming", "max_seq_len", "num_workers"],
    "logging_cfg": ["exp_name", "log_every_n_steps", "ckpt_every_n_epochs", "test_every_n_epochs", "log_dir", "ckpt_dir"],
}


def add_model_args(parser: argparse.ArgumentParser) -> None:
    # Training configs
    training_group = parser.add_argument_group("Training configs")
    training_group.add_argument("--max_epochs", type=int, help="maximum number of epochs to train")
    training_group.add_argument("--batch_size", type=int, help="number of instances, i.e., number of videos in a batch of data")
    training_group.add_argument("--seed", type=int_or_none, help="random seed (default: 42)")
    training_group.add_argument("--pretrained_ckpt", type=str_or_none, help="path to pretrained checkpoint (default: None)")

    # Optimizer configs
    optimizer_group = parser.add_argument_group("Optimizer configs")
    optimizer_group.add_argument("--lr", type=float, help="Learning rate.")
    optimizer_group.add_argument("--lr_scheduler.milestones", type=int, nargs="+", help="LR scheduler milestones.")
    optimizer_group.add_argument("--lr_scheduler.gamma", type=float, help="LR scheduler gamma.")

    # Model configs
    model_group = parser.add_argument_group("Model Configs")
    model_group.add_argument("--dropout_prob", type=float, help="Dropout probability.")
    model_group.add_argument("--hlc_ctx_len", type=int, help="HLC context length.")
    model_group.add_argument("--threshold", type=float, help="Threshold value.")
    model_group.add_argument("--sigma", type=float, help="Sigma value.")
    model_group.add_argument("--gamma", type=float, help="Gamma value.")
    model_group.add_argument("--loss.lambda", type=float, help="Lambda value.")
    model_group.add_argument("--loss.is_topk", type=str2bool, nargs="?", const=True, help="Flag for using top-k pooling.")
    model_group.add_argument("--loss.q", type=int, help="Q value.")

    # Dataset configs
    dataset_group = parser.add_argument_group("Dataset Configs")
    dataset_group.add_argument("--feature_name", type=str, help="Feature name.")
    dataset_group.add_argument(
        "--seperated_by_class", type=str2bool, nargs="?", const=True, help="Whether to create individual datasets for each class."
    )
    dataset_group.add_argument("--feature_dim", type=int, help="Feature dimension.")
    dataset_group.add_argument("--clip_len", type=int, help="Clip length.")
    dataset_group.add_argument("--sampling_rate", type=int, help="Sampling rate.")
    dataset_group.add_argument("--streaming", type=str2bool, nargs="?", const=True, help="Streaming mode flag.")
    dataset_group.add_argument("--max_seq_len", type=int, help="Maximum sequence length.")
    dataset_group.add_argument("--num_workers", type=int_or_none, help="Number of workers for data loading.")

    # Logging configs
    logging_group = parser.add_argument_group("Logging Configs")
    logging_group.add_argument("--exp_name", type=str_or_none, help="Experiment name.")
    logging_group.add_argument("--log_every_n_steps", type=int, help="Log every N steps.")
    logging_group.add_argument("--ckpt_every_n_epochs", type=int_or_none, help="Checkpoint every N epochs.")
    logging_group.add_argument("--test_every_n_epochs", type=int, help="Test every N epochs.")
    logging_group.add_argument("--log_dir", type=str_or_none, help="Log directory.")
    logging_group.add_argument("--ckpt_dir", type=str_or_none, help="Checkpoint directory.")
