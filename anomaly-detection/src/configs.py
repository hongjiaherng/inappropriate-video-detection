import argparse

DEFAULT_MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42
DEFAULT_EXP_NAME = "default"
DEFAULT_PRETRAINED_CKPT = None
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_SEQ_LEN = 100
DEFAULT_STREAMING = False
DEFAULT_FEATURE_NAME = "i3d_rgb"
DEFAULT_FEATURE_DIM = 2048


def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)


def parse_configs():
    parser = argparse.ArgumentParser(
        description="Temporal Anomaly Detection in Video with Weak Supervision"
    )

    # Training configs
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="maximum number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="number of instances, i.e., number of videos in a batch of data (default: 128)",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR, help="learning rate (default: 1e-3)"
    )

    # Experiemnt configs
    parser.add_argument(
        "--exp_name",
        type=str,
        default=DEFAULT_EXP_NAME,
        help="experiment name (default: default)",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=DEFAULT_PRETRAINED_CKPT,
        help="path to pretrained checkpoint (default: None)",
    )

    # Data loading configs
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=DEFAULT_STREAMING,
        help="whether to stream the dataset in the loop or download it all at once before starting (default: False)",
    )
    parser.add_argument(
        "--feature_name",
        type=str,
        default=DEFAULT_FEATURE_NAME,
        help="feature name (default: i3d_rgb)",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=DEFAULT_FEATURE_DIM,
        help="feature dimension (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int_or_none,
        default=DEFAULT_SEED,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--num_workers",
        type=int_or_none,
        default=DEFAULT_NUM_WORKERS,
        help="number of workers for data loading (default: 4)",
    )

    # Hyperparameters
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="maximum sequence length, i.e., total number of clips per video (default: 100)",
    )

    return vars(parser.parse_args())
