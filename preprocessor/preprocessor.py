import os
import random
import json
import re
import time

import tgt
import librosa
import numpy as np
# import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

import audio as Audio
from model import PreDefinedEmbedder
from utils.pitch_tools import get_pitch, get_cont_lf0, get_lf0_cwt
from utils.tools import plot_embedding
import multiprocessing as mp


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        self.in_dir = preprocess_config["path"]["raw_path"]
        #self.corpus_dir = preprocess_config["path"]["corpus_path"]
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.multi_speaker = model_config["multi_speaker"]

        self.with_f0 = preprocess_config["preprocessing"]["pitch"]["with_f0"]
        self.with_f0cwt = preprocess_config["preprocessing"]["pitch"]["with_f0cwt"]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.energy_phoneme_averaging = (
            preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        ).cuda()
        self.val_prior = self.val_prior_names(os.path.join(self.out_dir, "val.txt"))
        self.speaker_emb = None
        self.in_sub_dirs = [p for p in os.listdir(self.in_dir) if os.path.isdir(os.path.join(self.in_dir, p))]
        if self.multi_speaker and preprocess_config["preprocessing"]["speaker_embedder"] != "none":
            self.speaker_emb = PreDefinedEmbedder(preprocess_config).cuda()
            self.speaker_emb_dict = self._init_spker_embeds(self.in_sub_dirs)

    def _init_spker_embeds(self, spkers):
        spker_embeds = dict()
        for spker in spkers:
            spker_embeds[spker] = list()
        return spker_embeds

    def val_prior_names(self, val_prior_path):
        val_prior_names = set()
        if os.path.isfile(val_prior_path):
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                for m in f.readlines():
                    val_prior_names.add(m.split("|")[0])
            return list(val_prior_names)
        else:
            return None

    def process_utterance(self, speaker, basename, save_speaker_emb):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename))

        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        mel2ph_filename = "{}-mel2ph-{}.npy".format(speaker, basename)
        f0_filename = "{}-f0-{}.npy".format(speaker, basename)
        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        cwt_spec_filename = "{}-cwt_spec-{}.npy".format(speaker, basename)
        cwt_scales_filename = "{}-cwt_scales-{}.npy".format(speaker, basename)
        f0cwt_mean_std_filename = "{}-f0cwt_mean_std-{}.npy".format(speaker, basename)
        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)

        array_paths = [
            os.path.join(self.out_dir, "duration", dur_filename),
            os.path.join(self.out_dir, "mel2ph", mel2ph_filename),
            os.path.join(self.out_dir, "f0", f0_filename),
            os.path.join(self.out_dir, "pitch", pitch_filename),
            os.path.join(self.out_dir, "cwt_spec", cwt_spec_filename),
            os.path.join(self.out_dir, "cwt_scales", cwt_scales_filename),
            os.path.join(self.out_dir, "f0cwt_mean_std", f0cwt_mean_std_filename),
            os.path.join(self.out_dir, "energy", energy_filename),
            os.path.join(self.out_dir, "mel", mel_filename)
        ]




        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, mel2ph, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        spker_embed = self.speaker_emb(wav) if save_speaker_emb else None
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        processed = True
        for path in array_paths:
            processed = processed and os.path.isfile(path)
        if processed:
            mel_spectrogram = np.load(array_paths[-1]).T
            return (
                "|".join([basename, speaker, text, raw_text]),
                np.load(array_paths[2]),
                self.remove_outlier(np.load(array_paths[-2])),
                mel_spectrogram.shape[1],
                np.min(mel_spectrogram, axis=1),
                np.max(mel_spectrogram, axis=1),
                spker_embed,
            )


        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Compute pitch
        if self.with_f0:
            f0, pitch = self.get_pitch(wav, mel_spectrogram.T)
            if f0 is None or sum(f0) == 0:
                return None
            if self.with_f0cwt:
                cwt_spec, cwt_scales, f0cwt_mean_std = self.get_f0cwt(f0)
                if np.any(np.isnan(cwt_spec)):
                    return None

        # Save files
        arrays = [
            duration,
            mel2ph,
            f0,
            pitch,
            cwt_spec,
            cwt_scales,
            f0cwt_mean_std,
            energy,
            mel_spectrogram.T
        ]
        for path, array in zip(array_paths, arrays):
            np.save(path, array)

        return (
            "|".join([basename, speaker, text, raw_text]),
            f0,
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
            np.min(mel_spectrogram, axis=1),
            np.max(mel_spectrogram, axis=1),
            spker_embed,
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        mel2ph = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        # Get mel2ph
        for ph_idx in range(len(phones)):
            mel2ph += [ph_idx + 1] * durations[ph_idx]
        assert sum(durations) == len(mel2ph)

        return phones, durations, mel2ph, start_time, end_time

    def get_pitch(self, wav, mel):
        f0, pitch_coarse = get_pitch(wav, mel, self.preprocess_config)
        return f0, pitch_coarse

    def get_f0cwt(self, f0):
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        logf0s_mean_std_org = np.array([logf0s_mean_org, logf0s_std_org])
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        return Wavelet_lf0, scales, logf0s_mean_std_org

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

    def divide_speaker_by_gender(self, in_dir, speaker_path="speaker-info.txt"):
        speakers = dict()
        with open(os.path.join(in_dir, speaker_path), encoding='utf-8') as f:
            for line in tqdm(f):
                if "ID" in line: continue
                parts = [p.strip() for p in re.sub(' +', ' ',(line.strip())).split(' ')]
                spk_id, gender = parts[0], parts[2]
                speakers[str(spk_id)] = gender
        return speakers

    def load_embedding(self, embedding_dir):
        embedding_path_list = [_ for _ in Path(embedding_dir).rglob('*.npy')]
        embedding = None
        embedding_speaker_id = list()
        # Gather data
        for path in tqdm(embedding_path_list):
            embedding = np.concatenate((embedding, np.load(path)), axis=0) \
                                            if embedding is not None else np.load(path)
            embedding_speaker_id.append(str(str(path).split('/')[-1].split('-')[0]))
        return embedding, embedding_speaker_id

def utterance_process(preprocess_config, model_config, train_config, task_queue, ret_queue, pid):
    processor = Preprocessor(preprocess_config, model_config, train_config)
    while True:
        if task_queue.empty():
            continue
        msg = task_queue.get()
        print(msg)
        speaker, basename, save_speaker_emb = msg
        ret = processor.process_utterance(speaker, basename, save_speaker_emb)
        ret_queue.put((msg, ret))

def try_utterance_process(preprocess_config, model_config, train_config, task_queue, ret_queue, pid):
    import traceback
    try:
        utterance_process(preprocess_config, model_config, train_config, task_queue, ret_queue, pid)
    except Exception as e:
        print('Process Exception: ', e)
        print(traceback.format_exc())
        raise e

def normalize(in_dir, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        np.save(filename, values)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value

def build_from_path(preprocess_config, model_config, train_config):
    in_dir = preprocess_config["path"]["raw_path"]
    out_dir = preprocess_config["path"]["preprocessed_path"]
    val_size = preprocess_config["preprocessing"]["val_size"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    speaker_emb = model_config["multi_speaker"] and preprocess_config["preprocessing"]["speaker_embedder"] != "none"
    if speaker_emb:
        spkers = [p for p in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, p))]
        speaker_emb_dict = dict()
        for spker in spkers:
            speaker_emb_dict[spker] = list()
    val_prior_path = os.path.join(out_dir, "val.txt")
    val_prior = None
    if os.path.isfile(val_prior_path):
        val_prior = set()
        print("Load pre-defined validation set...")
        with open(val_prior_path, "r", encoding="utf-8") as f:
            for m in f.readlines():
                val_prior.add(m.split("|")[0])
        val_prior = list(val_prior)

    embedding_dir = os.path.join(out_dir, "spker_embed")
    os.makedirs((os.path.join(out_dir, "mel")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "f0")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "pitch")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "cwt_spec")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "cwt_scales")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "f0cwt_mean_std")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "energy")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "duration")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "mel2ph")), exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    task_queue, ret_queue = mp.Queue(), mp.Queue()
    num_processes = mp.cpu_count()
    processes = [mp.Process(target=try_utterance_process, args=(preprocess_config, model_config, train_config, task_queue, ret_queue, n))
                 for n in range(num_processes)]
    for p in processes:
        p.daemon = True
        p.start()
    print("Processing Data ...")
    filtered_out = set()
    out = list()
    train = list()
    val = list()
    n_frames = 0
    max_seq_len = -float('inf')
    mel_min = np.ones(n_mel_channels) * float('inf')
    mel_max = np.ones(n_mel_channels) * -float('inf')
    f0s = []
    energy_scaler = StandardScaler()

    skip_speakers = set()
    for embedding_name in os.listdir(embedding_dir):
        skip_speakers.add(embedding_name.split("-")[0])

    # Compute pitch, energy, duration, and mel-spectrogram
    speakers = {}
    task_set = set()
    for i, speaker in enumerate(tqdm(os.listdir(in_dir))):
        save_speaker_emb = speaker_emb and speaker not in skip_speakers
        speakers[speaker] = i
        for wav_name in tqdm(os.listdir(os.path.join(in_dir, speaker))):
            if ".wav" not in wav_name:
                continue

            basename = wav_name.split(".")[0]
            tg_path = os.path.join(
                out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
            )
            if os.path.exists(tg_path):
                msg = (speaker, basename, save_speaker_emb)
                task_set.add(msg)
                task_queue.put(msg)
    while ret_queue.empty():
        pass
    t = time.time()
    while task_set:
        if ret_queue.empty():
            if time.time() < t+1:
                time.sleep(1)
            else:
                msg = task_set.pop()
                task_set.add(msg)
                task_queue.put(msg)
            t = time.time()
            continue
        msg, ret = ret_queue.get()
        if msg not in task_set:
            continue
        task_set.remove(msg)
        speaker, basename, save_speaker_emb = msg
        if ret is None:
            filtered_out.add(basename)
            continue
        info, f0, energy, n, m_min, m_max, spker_embed = ret
        if val_prior is not None:
            if basename not in val_prior:
                train.append(info)
            else:
                val.append(info)
        else:
            out.append(info)

        if len(f0) > 0:
            f0s.append(f0)
        if len(energy) > 0:
            energy_scaler.partial_fit(energy.reshape((-1, 1)))

        if save_speaker_emb:
            speaker_emb_dict[speaker].append(spker_embed)

        mel_min = np.minimum(mel_min, m_min)
        mel_max = np.maximum(mel_max, m_max)

        if n > max_seq_len:
            max_seq_len = n

        n_frames += n

    print("speaker_emb", speaker_emb)
        # Calculate and save mean speaker embedding of this speaker
    if speaker_emb:
        for speaker, embeds in speaker_emb_dict.items():
            if not embeds:
                continue
            spker_embed_path = os.path.join(out_dir, 'spker_embed', '{}-spker_embed.npy'.format(speaker))
            np.save(spker_embed_path, np.mean(embeds, axis=0), allow_pickle=False)

    print("Computing statistic quantities ...")
    if len(f0s) > 0:
        f0s = np.concatenate(f0s, 0)
        f0s = f0s[f0s != 0]
        f0_mean = np.mean(f0s).item()
        f0_std = np.std(f0s).item()

    # Perform normalization if necessary
    if preprocess_config["preprocessing"]["energy"]["normalization"]:
        energy_mean = energy_scaler.mean_[0]
        energy_std = energy_scaler.scale_[0]
    else:
        # A numerical trick to avoid normalization...
        energy_mean = 0
        energy_std = 1

    energy_min, energy_max = normalize(
        os.path.join(out_dir, "energy"), energy_mean, energy_std
    )

    # Save files
    with open(os.path.join(out_dir, "speakers.json"), "w") as f:
        f.write(json.dumps(speakers))

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        stats = {
            "f0": [
                float(f0_mean),
                float(f0_std),
            ],
            "energy": [
                float(energy_min),
                float(energy_max),
                float(energy_mean),
                float(energy_std),
            ],
            "spec_min": mel_min.tolist(),
            "spec_max": mel_max.tolist(),
            "max_seq_len": max_seq_len,
        }
        f.write(json.dumps(stats))

    print(
        "Total time: {} hours".format(
            n_frames * hop_length / sampling_rate / 3600
        )
    )

    """if self.speaker_emb is not None:
        print("Plot speaker embedding...")
        plot_embedding(
            self.out_dir, *self.load_embedding(embedding_dir),
            self.divide_speaker_by_gender(self.corpus_dir), filename="spker_embed_tsne.png"
        )"""

    filtered_out = list(filtered_out)
    if val_prior is not None:
        assert len(out) == 0
        random.shuffle(train)
        train = [r for r in train if r is not None]
        val = [r for r in val if r is not None]
    else:
        assert len(train) == 0 and len(val) == 0
        random.shuffle(out)
        out = [r for r in out if r is not None]
        train = out[val_size:]
        val = out[: val_size]

    # Write metadata
    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
        for m in train:
            f.write(m + "\n")
    with open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8") as f:
        for m in val:
            f.write(m + "\n")
    with open(os.path.join(out_dir, "filtered_out.txt"), "w", encoding="utf-8") as f:
        for m in sorted(filtered_out):
            f.write(str(m) + "\n")

    return out