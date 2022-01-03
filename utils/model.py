import os
import json

import torch
import numpy as np

import hifigan
import hifigan_2
from model import CompTransTTS, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = CompTransTTS(preprocess_config, model_config, train_config).to(device)
    if args.restore_step:
        if train_config["finetune_single_voice"]["finetune_single_voice"] is True:
            ckpt_path = train_config["finetune_single_voice"]["pretrained_ckpt_path"]
            print("Finetuning {}".format(train_config["finetune_single_voice"]["speaker_finetune"]))
        else:
            ckpt_path = os.path.join(
                train_config["path"]["ckpt_path"],
                "{}.pth.tar".format(args.restore_step),
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    denoise = config["vocoder"]["denoiser"]
    print(speaker, "vocoder")
    if name == "HiFi-GAN" and denoise is True:
        print("Using denoiser")
    else:
        print("Not using denoiser")

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        if denoise is True:
            from hifigan_2.denoiser import Denoiser as HFDenoiser
            from hifigan_2.inference_e2e import HifiGanInferenceE2E
            # with open("hifigan_2/{}/config.json".format(speaker), "r") as f:
            #     config = json.load(f)
            # config = hifigan.AttrDict(config)
            # vocoder = hifigan.Generator(config)
            # ckpt = torch.load("hifigan_2/{}/model".format(speaker), map_location=device)
            vocoder = HifiGanInferenceE2E("hifigan_2/{}/model".format(speaker), "hifigan_2/{}/config.json".format(speaker), device)
            denoiser = HFDenoiser(vocoder.generator, device=device)
        else:
            # with open("hifigan/NgocHuyenHifiGan/config.json", "r") as f:
            #     config = json.load(f)
            # config = hifigan.AttrDict(config)
            # vocoder = hifigan.Generator(config)
            if speaker == "LJSpeech":
                with open("hifigan/NgocHuyenHifiGan/config.json", "r") as f:
                    config = json.load(f)
                config = hifigan.AttrDict(config)
                vocoder = hifigan.Generator(config)
                ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
            elif speaker == "universal":
                with open("hifigan/NgocHuyenHifiGan/config.json", "r") as f:
                    config = json.load(f)
                config = hifigan.AttrDict(config)
                vocoder = hifigan.Generator(config)
                ckpt = torch.load("hifigan/NgocHuyenHifiGan/model", map_location=device)
            else:
                with open("hifigan_2/{}/config.json".format(speaker), "r") as f:
                    config = json.load(f)
                config = hifigan.AttrDict(config)
                vocoder = hifigan.Generator(config)
                ckpt = torch.load("hifigan_2/{}/model".format(speaker), map_location=device)
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
            denoiser = None
    # if denoise is True:
    #     from hifigan_2.denoiser import Denoiser as HFDenoiser
    #     denoiser = HFDenoiser(vocoder, device=device)
    # else:
    #     denoiser = None
    return vocoder, denoiser


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
