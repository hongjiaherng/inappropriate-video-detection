import argparse
import logging

import pengwu_net.inferer
import sultani_net.inferer


def get_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("infer")
    return logger


def get_inferece_parser_args():
    parser = argparse.ArgumentParser(description="Run Inference for Temporal Anomaly Detection in Video")
    parser.add_argument("--model_name", type=str, required=True, choices=["pengwu_net", "sultani_net", "svm_baseline"], help="Model name")
    parser.add_argument("--run_path", type=str, required=True, help="Wandb's run path for the trained model.")
    parser.add_argument("--ckpt_type", type=str, required=True, choices=["best", "last"], help="Use best or last checkpoint.")

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = get_inferece_parser_args()
    logger = get_logger()

    model_name = args.get("model_name", None)

    if model_name == "pengwu_net":
        pengwu_net.inferer.run(**args, logger=logger)
    elif model_name == "sultani_net":
        sultani_net.inferer.run(**args, logger=logger)
    elif model_name == "svm_baseline":
        pass
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
