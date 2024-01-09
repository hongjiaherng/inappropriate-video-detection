import logging

import baseconfig
import pengwu_net.runner
import sultani_net.runner
import svm_baseline.runner


PROJECT_NAME = "wsvad"


def get_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train")
    return logger


if __name__ == "__main__":
    model_name, config = baseconfig.get_parser_args()
    logger = get_logger()

    if model_name == "pengwu_net":
        pengwu_net.runner.run(**config, logger=logger, project_name=PROJECT_NAME)
    elif model_name == "sultani_net":
        sultani_net.runner.run(**config, logger=logger, project_name=PROJECT_NAME)
    elif model_name == "svm_baseline":
        svm_baseline.runner.run(**config, logger=logger, project_name=PROJECT_NAME)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
