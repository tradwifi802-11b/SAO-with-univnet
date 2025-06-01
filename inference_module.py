import os
import glob
import tqdm
import torch
from scipy.io.wavfile import write
from omegaconf import OmegaConf

from model.generator import Generator

def inference(
    checkpoint_path,
    input_folder,
    output_folder=None,
    config_path=None
):
    checkpoint = torch.load(checkpoint_path)
    if config_path is not None:
        hp = OmegaConf.load(config_path)
    else:
        hp = OmegaConf.create(checkpoint['hp_str'])

    model = Generator(hp).cuda()
    saved_state_dict = checkpoint['model_g']
    new_state_dict = {}

    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict['module.' + k]
        except:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval(inference=True)

    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(input_folder, '*.mel'))):
            mel = torch.load(melpath)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()

            audio = model.inference(mel)
            audio = audio.cpu().detach().numpy()

            if output_folder is None:
                out_path = melpath.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
            else:
                basename = os.path.basename(melpath)
                basename = basename.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
                out_path = os.path.join(output_folder, basename)
            write(out_path, hp.audio.sampling_rate, audio)

    print("Inference completed.")

