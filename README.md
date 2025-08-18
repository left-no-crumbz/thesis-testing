# TrueVoice
A repository for training the model for TrueVoice - a model for classifying audio deepfakes using the Audio Spectrogram Transformer (AST).

## Datasets used for training
- [WaveFake](https://zenodo.org/records/5642694)
- [In-the-Wild](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa)
- [Fake-or-Real](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- [Common Voices Mozilla](https://www.kaggle.com/datasets/mozillaorg/common-voice)
- [LibreSpeech](https://www.openslr.org/12)

## Datasets used for evaluating
- Our dataset's test split
- [LibriSeVoc](https://drive.google.com/file/d/1Zh6b51S1WIsFjdCDRTQhYW61CQ0Ue1lk/view)
- [ASVspoof2019 LA](https://www.kaggle.com/datasets/hahunavth/asvspoof2019-la)

## Important Files
- `app.ipynb` - Training script
- `model.ipynb` - Evaluation script
