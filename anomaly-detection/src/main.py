import logging

import pengwu_net.runner
import baseconfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    model_name, config = baseconfig.get_parser_args()

    if model_name == "pengwu_net":
        pengwu_net.runner.run(**config, logger=logger)
    elif model_name == "svm_baseline":
        pass
    else:
        raise ValueError(f"Invalid model name: {model_name}")
