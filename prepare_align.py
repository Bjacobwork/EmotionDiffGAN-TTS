import argparse

from utils.tools import get_configs_of
from preprocessor import ljspeech, vctk, eemotion, crema_d, jl_corpus

def main(dataset, config):
    if "Combined" in dataset:
        for dataset, path in config["path"]["corpus_paths"].items():
            config['path']['corpus_path'] = path
            main(dataset, config)
    if "LJSpeech" in dataset:
        ljspeech.prepare_align(config)
    if "VCTK" in dataset:
        vctk.prepare_align(config)
    if "EEmotion" in dataset:
        eemotion.prepare_align(config)
    if "CREMAD" in dataset:
        crema_d.prepare_align(config)
    if "JLCorpus" in dataset:
        jl_corpus.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    config, *_ = get_configs_of(args.dataset)
    main(config["dataset"], config)
