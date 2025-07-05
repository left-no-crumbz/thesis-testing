from datasets import load_dataset
from transformers import ASTFeatureExtractor
import torch
import numpy as np

dataset = load_dataset("audiofolder", data_dir="./for-2seconds")

pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model, num_mel_bins=64)

model_input_name = feature_extractor.model_input_names[0]
SAMPLING_RATE = feature_extractor.sampling_rate

print(model_input_name)
print(SAMPLING_RATE)


# calculate values for normalization
feature_extractor.do_normalize = False  # we set normalization to False in order to calculate the mean + std of the dataset
mean = []
std = []

def preprocess_audio(batch):
    wavs = [audio["array"] for audio in batch["input_values"]]
    # inputs are spectrograms as torch.tensors now
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    print(batch)
    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["label"])}
    return output_batch

dataset = dataset.rename_column("audio", "input_values") 
# we use the transformation w/o augmentation on the training dataset to calculate the mean + std
dataset["train"].set_transform(preprocess_audio, output_all_columns=False)

for i, (audio_input, labels) in enumerate(dataset["train"]):
    cur_mean = torch.mean(dataset["train"][i][audio_input])
    cur_std = torch.std(dataset["train"][i][audio_input])
    mean.append(cur_mean)
    std.append(cur_std)

feature_extractor.mean = np.mean(mean)
feature_extractor.std = np.mean(std)
feature_extractor.do_normalize = True

