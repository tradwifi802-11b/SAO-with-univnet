import os
import glob
import tqdm
import librosa
import numpy as np
import torch
from pydub import AudioSegment
import matplotlib.pyplot as plt  # <--- import matplotlib

INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'mel'
IMAGE_FOLDER = 'spectogram'   # folder for mel spectrogram images
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def convert_mp3_to_wav(mp3_path, wav_path, target_sr=24000):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)  # mono, 24kHz, 16-bit
    audio.export(wav_path, format='wav')

def compute_mel(wav_path, sr=24000, n_mels=100, fmin=0, fmax=12000):
    y, sr = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    mel = np.log(mel + 1e-9)
    return torch.from_numpy(mel)

def save_mel_image(mel_tensor, save_path):
    mel = mel_tensor.numpy()
    plt.figure(figsize=(10, 4))
    plt.title('Mel Spectrogram')
    plt.xlabel('Time frames')
    plt.ylabel('Mel frequency bins')
    plt.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# for filepath in tqdm.tqdm(glob.glob(os.path.join(INPUT_FOLDER, '*'))):
#     filename = os.path.basename(filepath)
#     name, ext = os.path.splitext(filename)
#     ext = ext.lower()

#     # mp3 to wav conversion
#     if ext == '.mp3':
#         wav_path = os.path.join(INPUT_FOLDER, f'{name}.wav')
#         convert_mp3_to_wav(filepath, wav_path)
#     elif ext == '.wav':
#         wav_path = filepath
#     else:
#         print(f"Skipping unsupported file format: {filename}")
#         continue

#     try:
#         mel_tensor = compute_mel(wav_path)
#         mel_save_path = os.path.join(OUTPUT_FOLDER, f'{name}.mel')
#         torch.save(mel_tensor, mel_save_path)

#         # Save mel image
#         image_save_path = os.path.join(IMAGE_FOLDER, f'{name}_mel.png')
#         save_mel_image(mel_tensor, image_save_path)

#     except Exception as e:
#         print(f"Failed to process {filename}: {e}")

def process_file(filepath):
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    ext = ext.lower()

    # mp3 to wav conversion
    if ext == '.mp3':
        wav_path = os.path.join(INPUT_FOLDER, f'{name}.wav')
        convert_mp3_to_wav(filepath, wav_path)
    elif ext == '.wav':
        wav_path = filepath
    else:
        print(f"Skipping unsupported file format: {filename}")
        return

    try:
        mel_tensor = compute_mel(wav_path)
        mel_save_path = os.path.join(OUTPUT_FOLDER, f'{name}.mel')
        torch.save(mel_tensor, mel_save_path)

        image_save_path = os.path.join(IMAGE_FOLDER, f'{name}_mel.png')
        save_mel_image(mel_tensor, image_save_path)

        print(f"Processed: {filename}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")


# for filepath in tqdm.tqdm(glob.glob(os.path.join(INPUT_FOLDER, '*'))):
#     process_file(filepath)
