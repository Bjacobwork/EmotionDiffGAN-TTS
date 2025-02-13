import argparse

from utils.tools import get_configs_of
from preprocessor.preprocessor import build_from_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    build_from_path(preprocess_config, model_config, train_config)
