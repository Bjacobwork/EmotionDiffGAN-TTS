from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
import os
from tqdm import tqdm
import librosa
import torch
import argparse
from utils.tools import get_configs_of


device = 0
model_path = "Booberjacob/wav2vec2-lg-xlsr-en-speech-circumplex-emotion-recognition"
wav2vec2 = AutoModelForAudioClassification.from_pretrained(model_path)
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
wav2vec2.to(device).eval()

def label_dataset(in_dir, out_dir, sequence_length=80000):
    stride_ratio = 4
    stride = sequence_length // stride_ratio
    os.makedirs(os.path.join(out_dir, "emotion"), exist_ok=True)
    for i, speaker in enumerate(tqdm(os.listdir(in_dir))):
        for wav_name in tqdm(os.listdir(os.path.join(in_dir, speaker))):
            if ".wav" not in wav_name:
                continue
            basename = wav_name.split(".")[0]
            tg_path = os.path.join(
                out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
            )
            if not os.path.exists(tg_path):
                continue
            array_path = os.path.join(out_dir, "emotion", f"{speaker}-emotion-{basename}.npy")
            wav_path = os.path.join(in_dir, speaker, "{}.wav".format(basename))
            wav, _ = librosa.load(wav_path, sr=16000)
            audio = torch.Tensor(wav)
            if len(audio.shape) > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            else:
                audio = audio.view(1, -1)
            if audio.shape[-1] > sequence_length:
                audio = torch.concat([audio[:, i * stride:i * stride + sequence_length] for i in
                                      range(audio.shape[-1] // stride - stride_ratio + 1)], 0)
            tensor_data = []
            for snippet in audio:
                tensor_data.append(extractor(raw_speech=snippet,
                              sampling_rate=16000,
                              padding=True,
                              return_tensors="pt"))
            length = 0
            for _input in tensor_data:
                length = max(length, _input['input_values'].shape[-1])
            batched_inputs = {key: [] for key in tensor_data[0]}
            for _input in tensor_data:
                for key in _input:
                    batched_inputs[key].append(torch.concat(
                        [_input[key], torch.zeros((*_input[key].shape[:-1], length - _input[key].shape[-1]))], -1))
            for key in batched_inputs:
                batched_inputs[key] = torch.concat(batched_inputs[key], 0).to(device)

            with torch.no_grad():
                emotions = wav2vec2(**batched_inputs)['logits'].cpu().numpy()
            np.save(array_path, np.mean(emotions, 0))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()
    preprocess_config, _, _ = get_configs_of(args.dataset)
    label_dataset(preprocess_config['path']['raw_path'],
                  preprocess_config['path']['preprocessed_path'])