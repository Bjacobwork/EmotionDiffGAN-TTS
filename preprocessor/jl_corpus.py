import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from text import _clean_text


def prepare_align(config):
    in_dir = os.path.join(config["path"]["corpus_path"],
                          "Raw JL corpus (unchecked and unannotated)",
                          "JL(wav+txt)")
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    for filename in tqdm(os.listdir(in_dir)):
        out_path = os.path.join(out_dir, f"JLCorpus_{filename.split('_')[0]}")
        os.makedirs(out_path, exist_ok=True)
        if filename.endswith(".wav"):
            wav, _ = librosa.load(os.path.join(in_dir, filename), sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_path, filename),
                sampling_rate,
                wav.astype(np.int16),
            )
        elif filename.endswith(".txt"):
            with open(os.path.join(in_dir, filename), "r") as file:
                text = _clean_text(file.read(), cleaners)
            with open(os.path.join(out_path, f"{filename[:-3]}lab"), 'w') as file:
                file.write(text)
