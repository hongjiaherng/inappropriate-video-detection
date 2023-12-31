import argparse
from typing import Dict

import pengwu_net.config
import svm_baseline.config
import yaml
from deepmerge import always_merger

MODEL_ARGS = {
    "pengwu_net": pengwu_net.config.add_model_args,
    "svm_baseline": svm_baseline.config.add_model_args,
}


def get_parser_args():
    parser = argparse.ArgumentParser(description="Temporal Anomaly Detection in Video with Weak Supervision")
    parser.add_argument("--config_path", type=str, help="Path to config file")

    # Subparsers for each model
    subparsers = parser.add_subparsers(dest="model_name", help="Model name", required=True)
    for model_name, add_model_args in MODEL_ARGS.items():
        model_parser = subparsers.add_parser(model_name, help=f"Model-specific arguments for {model_name}")
        add_model_args(model_parser)

    args = parser.parse_args()

    if args.model_name == "pengwu_net":
        config_shape = pengwu_net.config.CONFIG_SHAPE
    elif args.model_name == "svm_baseline":
        config_shape = svm_baseline.config.CONFIG_SHAPE
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    return args.model_name, ConfigParser(config_shape=config_shape).parse(args)


class ConfigParser:
    def __init__(self, config_shape: Dict):
        self.config_shape = config_shape

    def parse(self, args: argparse.Namespace) -> Dict:
        cli_config, config_path = self._parse_as_dict(args)
        if not config_path:
            return cli_config

        yaml_config = self._parse_as_yaml(config_path)
        joined_config = self._merge_and_filter_none_values(yaml_config, cli_config)
        return joined_config

    def _parse_as_dict(self, args: argparse.Namespace) -> Dict:
        # Warn: This only support 1 level of nesting in args for now, e.g., --lr_scheduler.milestones or --loss.lambda
        flat_dict = vars(args)
        config_dict = {}
        for group_k, content in self.config_shape.items():
            config_dict[group_k] = {}
            for item in content:
                if isinstance(item, str):
                    config_dict[group_k][item] = flat_dict.get(item, None)
                elif isinstance(item, dict):
                    for subgroup_k, subcontent in item.items():
                        config_dict[group_k][subgroup_k] = {}
                        for subitem in subcontent:
                            config_dict[group_k][subgroup_k][subitem] = flat_dict.get(f"{subgroup_k}.{subitem}", None)
                else:
                    raise ValueError(f"Invalid item type: {type(item)}")

        return config_dict, flat_dict.get("config_path", None)

    def _parse_as_yaml(self, config_path: str) -> Dict:
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
                # TODO: Validate yaml config shape with self.config_shape
            return config_dict
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found: {config_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Error loading YAML file: {str(e)}") from e

    def _merge_and_filter_none_values(self, yaml_config: Dict, cli_config: Dict) -> Dict:
        # Override yaml config with any cli args that are not None
        def filter_none_values(d):
            if isinstance(d, dict):
                result = {k: filter_none_values(v) for k, v in d.items() if v is not None}
                return {k: v for k, v in result.items() if v is not None}
            else:
                return d

        filtered_config = filter_none_values(cli_config)
        return always_merger.merge(yaml_config, filtered_config)