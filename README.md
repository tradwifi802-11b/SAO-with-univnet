
# UnivNet + Continuous Unconditioned Generation Pipeline

**UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation**

This repository includes:

- The original UnivNet vocoder implementation (*Jang et al.* [UnivNet](https://arxiv.org/abs/2106.07889)).
- A complete end-to-end continuous unconditioned generation pipeline using pretrained Stable Audio Open (SAO) for waveform generation and your own trained UnivNet model for vocoding.

## What's New in This Version

We have extended the UnivNet vocoder implementation to support continuous unconditioned audio generation:

- We use pretrained Stable Audio Open Small (https://huggingface.co/stabilityai/stable-audio-open-small) to generate raw waveform.
- Generated audio is converted into mel-spectrograms.
- Trained UnivNet vocoder models (c16 or c32) are used to reconstruct waveform audio from mel-spectrograms.

This allows you to use your trained UnivNet model as a part of a full generative pipeline.

## Pipeline Overview

1. Audio Generation (Unconditioned)  
- We use Stable Audio Open (diffusion model) to generate raw waveform.  
- Generation is unconditioned (prompt="") but fully controllable via steps, sampler_type, and cfg_scale.

2. Mel Spectrogram Extraction  
- The generated waveform is converted to mel-spectrograms using the same parameters as UnivNet.

3. Waveform Reconstruction  
- The UnivNet vocoder takes these mel-spectrograms and reconstructs final audio using your own trained weights.

## Updated Dependencies

Please make sure the following additional dependencies are installed:

```bash
pip install stable-audio-tools einops torchaudio omegaconf
```

## Notes

- The mel extraction remains fully compatible with UnivNet, using:

```yaml
audio:
  n_mel_channels: 100
  filter_length: 1024
  hop_length: 256
  win_length: 1024
  sampling_rate: 24000
  mel_fmin: 0.0
  mel_fmax: 12000.0
```

- Both UnivNet-c16 and c32 models remain supported for vocoder training and inference.

## Pretrained UnivNet Models

Pretrained UnivNet models are still available here:

- UnivNet-c16: https://drive.google.com/file/d/1Iqw9T0rRklLsg-6aayNk6NlsLVHfuftv/view?usp=sharing
- UnivNet-c32: https://drive.google.com/file/d/1QZFprpvYEhLWCDF90gSl6Dpn0gonS_Rv/view?usp=sharing

## Research References

- Jang et al., UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation (https://arxiv.org/abs/2106.07889)
- StabilityAI, Stable Audio Open Small (https://huggingface.co/stabilityai/stable-audio-open-small)

## License

This code is licensed under BSD 3-Clause License.

We referred following codes and repositories:

- The overall structure of the repository is based on https://github.com/seungwonpark/melgan
- datasets/dataloader.py from https://github.com/NVIDIA/waveglow (BSD 3-Clause License)
- model/mpd.py from https://github.com/jik876/hifi-gan (MIT License)
- model/lvcnet.py from https://github.com/zceng/LVCNet (Apache License 2.0)
- utils/stft_loss.py Copyright 2019 Tomoki Hayashi (MIT License)