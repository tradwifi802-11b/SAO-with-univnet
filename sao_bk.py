import os
import argparse
import torch
import torchaudio
from einops import rearrange
from omegaconf import OmegaConf
from stable_audio_tools import get_pretrained_model as load_sao_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from model.generator import Generator
from scipy.io.wavfile import write
import numpy as np  # needed for the 1D→2D reshape check


def load_libritts_generator(config_path, checkpoint_path, device):
    """
    Loads the LibriTTS Generator model and returns:
      - generator (nn.Module)
      - hp (OmegaConf struct for audio/gen hyperparameters)
      - checkpoint (the raw checkpoint dict, so we can read checkpoint['epoch'])
    """
    hp = OmegaConf.load(config_path)

    print("Loaded LibriTTS config with keys:")
    print("  audio:", list(hp.audio.keys()))
    print("  gen:  ", list(hp.gen.keys()))

    print("== LibriTTS audio hyperparameters ==")
    print("sampling_rate:",        hp.audio.sampling_rate)
    print("filter_length (n_fft):", hp.audio.filter_length)
    print("hop_length:",           hp.audio.hop_length)
    print("win_length:",           hp.audio.win_length)
    print("n_mel_channels:",       hp.audio.n_mel_channels)
    print("mel_fmin, mel_fmax:",   hp.audio.mel_fmin, hp.audio.mel_fmax)
    # print("log_mel (True/False):", hp.audio.log_mel)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Determine where the generator state dict lives:
    if "model_g" in checkpoint:
        state_dict = checkpoint["model_g"]
        print("Loaded state_dict from checkpoint['model_g']")
    elif "generator" in checkpoint:
        state_dict = checkpoint["generator"]
        print("Loaded state_dict from checkpoint['generator']")
    else:
        state_dict = checkpoint
        print("Assuming entire checkpoint is the generator state_dict")

    # Remove any 'module.' prefix (if DataParallel was used)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k.replace("module.", "")
        else:
            new_key = k
        new_state_dict[new_key] = v

    generator = Generator(hp).to(device)
    generator.load_state_dict(new_state_dict)
    generator.remove_weight_norm()
    generator.eval()

    print("LibriTTS Generator loaded.")
    return generator, hp, checkpoint


def build_mel_transform(hp, device):
    """
    Creates a torchaudio.transforms.MelSpectrogram using the values from hp.audio.
    Falls back to hp.audio.win_length for n_fft if hp.audio.filter_length is invalid.
    """
    try:
        n_fft = int(hp.audio.filter_length)
    except (ValueError, TypeError):
        n_fft = int(hp.audio.win_length)
        print(f"Warning: filter_length invalid, using win_length={n_fft} for n_fft.")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(hp.audio.sampling_rate),
        n_fft=n_fft,
        hop_length=int(hp.audio.hop_length),
        win_length=int(hp.audio.win_length),
        f_min=float(hp.audio.mel_fmin),
        f_max=float(hp.audio.mel_fmax),
        n_mels=int(hp.audio.n_mel_channels),
        # Do NOT specify n_channels here; MelSpectrogram expects a 2D input (B, T)
    ).to(device)

    return mel_transform


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ───── 1) Load SAO Model ─────────────────────────────────────────────────────────
    print("Loading Stable Audio Open model...")
    sao_model, sao_config = load_sao_model(args.sao_ckpt)
    sao_model = sao_model.to(device).eval()

    sample_rate = sao_config["sample_rate"]
    sample_size = sao_config["sample_size"]

    # ───── 2) Load LibriTTS Generator + checkpoint dict ───────────────────────────────
    generator, hp, checkpoint = load_libritts_generator(
        args.config, args.checkpoint, device
    )

    # ───── 3) Generate SAO Audio ──────────────────────────────────────────────────────
    print("Generating SAO audio...")
    model_kwargs = dict(
        model=sao_model,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        sample_size=sample_size,
        sampler_type=args.sampler_type,
        device=device,
        conditioning=[{
            "prompt": args.prompt or " ",
            "seconds_total": float(args.seconds_total),
        }],
    )

    with torch.no_grad():
        audio_tensor = generate_diffusion_cond(**model_kwargs)
        # audio_tensor shape: (B, 1, N_samples)

    # ───── 4) Save SAO Output WAV ─────────────────────────────────────────────────────
    sao_wav = rearrange(audio_tensor, "b d n -> d (b n)")
    sao_wav = sao_wav.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    os.makedirs(args.output_dir, exist_ok=True)
    sao_path = os.path.join(args.output_dir, "sao_output.wav")
    torchaudio.save(sao_path, sao_wav, sample_rate)
    print(f"Saved: {sao_path}")

    # ───── 5) Convert SAO Audio → Mel Spectrogram ─────────────────────────────────────
    print("Converting SAO audio to mel spectrogram...")
    mel_transform = build_mel_transform(hp, device)

    # If SAO sample_rate differs from LibriTTS’s, resample first
    if sample_rate != int(hp.audio.sampling_rate):
        print(f"Resampling SAO audio from {sample_rate} to {hp.audio.sampling_rate}")
        resampler = torchaudio.transforms.Resample(
            sample_rate, int(hp.audio.sampling_rate)
        ).to(device)
        audio_tensor = resampler(audio_tensor)

    with torch.no_grad():
        # Ensure mono: if (B, 2, N) or more channels, average them
        if audio_tensor.ndim == 3 and audio_tensor.size(1) > 1:
            print("Converting stereo SAO audio to mono...")
            audio_wave = audio_tensor.mean(dim=1)  # → (B, N)
        else:
            audio_wave = audio_tensor.squeeze(1)   # → (B, N)

        # Compute Mel spectrogram: (B, n_mel_channels, T_frames)
        mel_spec = mel_transform(audio_wave)

        # Optional log‐Mel if your LibriTTS was trained on log‐Mel
        if getattr(hp.audio, "log_mel", False):
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        print("Final mel_spec shape:", mel_spec.shape)
        # e.g. torch.Size([1, 100, 1115])

    # ───── 6) Feed Mel into LibriTTS Generator ────────────────────────────────────────
    print("Feeding mel into LibriTTS vocoder...")
    print("mel_spec shape before inference:", mel_spec.shape)

    with torch.no_grad():
        # Most LibriTTS Generator.inference() variants expect a 3D mel: (B, n_mel, T)
        audio_gen = generator.inference(mel_spec)

        # If output is (B, 1, N_samples), squeeze channel:
        if audio_gen.ndim == 3 and audio_gen.size(1) == 1:
            audio_gen = audio_gen.squeeze(1)  # → (B, N_samples)

        audio_gen = audio_gen.cpu().numpy()

        # ───── NEW: ensure batch dimension exists ────────────────────────────────────
        if audio_gen.ndim == 1:
            # inference returned shape (N_samples,) → treat that as batch=1
            audio_gen = audio_gen[np.newaxis, :]
            print("Warning: inference returned a 1‑D array; reshaped to (1, N_samples).")

    # ───── 7) Save LibriTTS Outputs ───────────────────────────────────────────────────
    epoch_num = checkpoint.get("epoch", 0)
    for b in range(audio_gen.shape[0]):
        data = audio_gen[b]
        # If data is 1D (N_samples,), reshape to (N_samples, 1) for scipy.write
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        fname = os.path.join(
            args.output_dir,
            f"libritts_output_{b:02d}_epoch{epoch_num:04d}.wav"
        )
        write(fname, int(hp.audio.sampling_rate), data)
        print(f"Saved: {fname}")

    print("Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAO + LibriTTS Inference Pipeline"
    )

    # SAO arguments
    parser.add_argument(
        "--sao_ckpt", type=str, default="stabilityai/stable-audio-open-small",
        help="Name or path of SAO model"
    )

    # LibriTTS arguments
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to LibriTTS config YAML file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to LibriTTS vocoder .pt checkpoint"
    )

    # Generation arguments
    parser.add_argument(
        "--prompt", type=str, default="lo-fi piano melody",
        help="Prompt for SAO generation"
    )
    parser.add_argument(
        "--seconds_total", type=float, default=10.0,
        help="Length of generated audio (seconds)"
    )
    parser.add_argument(
        "--steps", type=int, default=8,
        help="Number of diffusion steps for SAO"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=1.0,
        help="Classifier-free guidance scale for SAO"
    )
    parser.add_argument(
        "--sampler_type", type=str, default="pingpong",
        help="Sampler type for SAO (e.g. 'pingpong')"
    )

    # Output directory
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory to save outputs"
    )

    args = parser.parse_args()
    main(args)

