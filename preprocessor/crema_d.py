import os
import audioread.exceptions
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from text import _clean_text
from pprint import pformat

TEXT = {'DFA': "Don't forget a jacket.",
 'IEO': "It's eleven o'clock.",
 'IOM': "I'm on my way to the meeting.",
 'ITH': "I think I have a doctor's appointment.",
 'ITS': "I think I've seen this before.",
 'IWL': 'I would like a new alarm clock.',
 'IWW': 'I wonder what this is about.',
 'MTI': 'Maybe tomorrow it will be cold.',
 'TAI': 'The airplane is almost full.',
 'TIE': 'That is exactly what happened.',
 'TSI': 'The surface is slick.',
 'WSI': "We'll stop in a couple of minutes."}

def prepare_align(config):
    #speaker = "CREMAD"
    in_dir = os.path.join(config["path"]["corpus_path"],
                          "AudioWAV")
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    ignore = ['1076_MTI_SAD_XX.wav', '1001_TAI_NEU_XX.wav']
    errors = []
    for filename in tqdm(os.listdir(in_dir)):
        if filename in ignore:
            continue
        line = filename.split("_")[1]
        if line not in TEXT:
            print(line, "- not in text")
            continue
        if not filename.endswith(".wav"):
            continue
        try:
            wav, _ = librosa.load(os.path.join(in_dir, filename), sampling_rate)
        except audioread.exceptions.NoBackendError as e:
            errors.append(filename)
            print(e)
            continue
        out_path = os.path.join(out_dir, f"CREMAD_{filename.split('_')[0]}")
        os.makedirs(out_path, exist_ok=True)
        wav = wav / max(abs(wav)) * max_wav_value
        wavfile.write(
            os.path.join(out_path, filename),
            sampling_rate,
            wav.astype(np.int16),
        )
        with open(os.path.join(out_path, f"{filename[:-3]}lab"), 'w') as file:
            file.write(_clean_text(TEXT[line], cleaners))
    print(f"ignored = {pformat(ignore+errors)}")