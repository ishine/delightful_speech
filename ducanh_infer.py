import sys
sys.path.append('..')

import torch
import yaml
import numpy as np
from scipy.io.wavfile import write

from hifigan.denoiser import Denoiser as HFDenoiser
from hifigan.inference_e2e import HifiGanInferenceE2E
from text import text_to_sequence
# from model.fastspeech2 import FastSpeech2
from model.fastspeech2_nguyenlm import FastSpeech2
from utils.tools import to_device, pad_1D

device = torch.device("cpu")

base_config_path = "config/NgocHuyenNews_fmax-8000"
prepr_path = f"{base_config_path}/preprocess.yaml"
model_path = f"{base_config_path}/model.yaml"
train_path = f"{base_config_path}/train.yaml"

prepr_config = yaml.load(open(prepr_path, "r"), Loader=yaml.FullLoader)
prepr_config["path"]["corpus_path"] = f'{prepr_config["path"]["corpus_path"]}'
prepr_config["path"]["raw_path"] = f'{prepr_config["path"]["raw_path"]}'
prepr_config["path"]["preprocessed_path"] = f'{prepr_config["path"]["preprocessed_path"]}'

model_config = yaml.load(open(model_path, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(train_path, "r"), Loader=yaml.FullLoader)
configs = (prepr_config, model_config, train_config)

# fastspeech checkpoint path
cpkt_path = "output/NgocHuyenNews_fmax-8000/ckpt/660000.pth.tar"

# hifigan model and config path
# base_hifi_path = "/home2/nguyenlm/Projects/TTS/deploy/deploy-fhg-change-speed/models/hn_female_ngochuyen_news_22k-fhg/V3.0/hifigan"
# hifi_model_path = f"{base_hifi_path}/model"
# hifi_conf_path  = f"{base_hifi_path}/config.json"

base_hifi_path = "./hifigan"
hifi_model_path = f"{base_hifi_path}/generator_universal.pth.tar"
hifi_conf_path  = f"{base_hifi_path}/config.json"

def get_ftsp_model(cpkt_path: str, configs: dict):

    def get_model(ckpt_path, configs):
        (preprocess_config, model_config, train_config) = configs
        model = FastSpeech2(preprocess_config, model_config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        model.requires_grad_ = False
        return model

    model = get_model(cpkt_path, configs)
    return model

def get_hifi_model(hifi_model_path: str, hifi_config_path: str):
    hifi_vocoder = HifiGanInferenceE2E(hifi_model_path, hifi_config_path)
    hifi_denoiser = HFDenoiser(hifi_vocoder.generator)
    return hifi_vocoder, hifi_denoiser

def get_sequence_from_text(input_text):
    cleaner_names = ["basic_cleaners"]
    sequence = text_to_sequence(input_text, cleaner_names)
    assert len(input_text.split()) == len(sequence), "[Warning]: Len senquence doesn't equal num of phonemes: phonemes {} - sequence {}".format(len(input_text.split()), len(sequence))
    return sequence


# get fastspeech model
model = get_ftsp_model(cpkt_path, configs)

# get hifigan vocoder and hifigan denoiser
hifi_vocoder, hifi_denoiser = get_hifi_model(hifi_model_path, hifi_conf_path)

def tts(phoneme_str, d_controls, p_controls, e_controls):
    sequences = [get_sequence_from_text(phoneme_str)]
    ids = raw_texts = ["validation"]
    speakers = np.array([0])
    texts = np.array([np.array(seq) for seq in sequences])
    texts = pad_1D(texts)
    text_lens = np.array([len(seq) for seq in sequences])
    batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
    batch = to_device(batch, device)

    d_controls = torch.tensor([d_controls])
    p_controls = torch.tensor([p_controls])
    e_controls = torch.tensor([e_controls])

    with torch.no_grad():
        # predict mel spectrogram
        output = model(
            *(batch[2:]),
            p_control=p_controls,
            e_control=e_controls,
            d_control=d_controls
        )
            
        mel_len = output[9][0].item()
        mel_prediction = output[1][0, :mel_len].detach().transpose(0, 1)

        # convert mel to waveform using hifigan
        hf_audio, _ = hifi_vocoder.inference(mel_prediction.unsqueeze(0), device)
        hf_audio_denoised = hifi_denoiser(hf_audio.unsqueeze(0), strength=0.01)[:, 0].squeeze().cpu().numpy()

    return hf_audio_denoised, output[5].squeeze(0).cpu().numpy().round().astype('int32')

if __name__ == "__main__":
    # 0 - ids,  1 - raw_texts, 2 - speakers, 3 - texts,
    # 4 - src_lens, 5 - max_src_len, 6 - mels, 7 - mel_lens,
    # 8 - max_mel_len, 9 - pitches, 10 - energies, 11 -durations
    import os
    m={'0000686': '{z 9X5 j5 # ts M5 N5 - J 9X7 n7 # d ie7 n7 - t M3 # k uo3 # b a7 n7 # d M98 k8 # G M3 j3 # v a2 w2 # s o5 # sp x o1 Nm1 # ts i5 n5 # x o1 Nm1 # ts i5 n5 # sp ts i5 n5 # t a5 m5 # t a5 m5 # sp h a1 j1 # h a1 j1 # h a1 j1 # sp i1 - m E1 w1 # sp h a1 - th i1 - M2 n2 - b a1 N1 # sp a1 - k OX2 Nm2 # G 92 - m E1 w1 # ts 9X5 m5 - k O1 m1 # sp v a2 # ts OX1 Nm1 # h o8 p8 - th M1 # k uo3 # M5 N5 - z u7 Nm7 # sp}', '0009948': '{ts i3 # k 9X2 n2 # d e3 # th uo1 # ts OX1 Nm1 # ts 9X7 n7 # ts u1 Nm1 - k e6 t6 # ts E1 m1 - p i1 - M2 n2 - s # l i6 k6 # ts M96 k6 # l i1 - v 91 - p u1 n1 # sp ts aX6 k6 - ts aX5 n5 # z i1 - d a1 n1 # l 9X8 p8 - t M6 k6 # b i7 # p e1 - r E6 t6 # d a5 - d i6 t6 # z a1 # d M92 N2 # sp}', '0011127': '{n e5 w5 # N M92 j2 - z 9X1 n1 # ts i7 w7 - x O5 # d i1 # s a1 # m o8 t8 - ts u6 t6 # s E4 # k O5 # ts o4 # G M3 j3 # v a2 # x i1 # t 9X8 p8 - ts u1 Nm1 # d i1 # t i2 m2 # sp ts i3 # m 9X6 t6 # m o8 t8 # N aX2 j2 # sp l a2 # t i2 m2 # d M98 k8 # b a4 j4 # d o4 # f u2 - h 98 p8 # sp}', '0009652': '{o1 Nm1 # k o1 - n o1 - n E1 n1 - k o1 # ts O1 # z aX2 N2 # sp J a2 - b a5 w5 # v i1 - S i1 n1 - s - k i1 # u3 Nm3 - h o7 # l M8 k8 - l M97 N7 # d o5 j5 - l 9X8 p8 # sp t a7 j7 # d o1 - n E6 t6 - s - k # v a2 # l u1 - G a1 n1 - s - k # 93 # m ie2 n2 # d o1 Nm1 # sp u6 kp6 - k - r a1 j1 - n a1 # sp}', '0008489': '{v ie8 k8 # d ie2 w2 - h EX2 J2 # ts EX5 J5 # k i2 m2 - n E5 n5 # z 9X4 n4 # t 95 j5 # EX6 k6 - t aX6 k6 # N uo2 n2 # k u1 Nm1 # ts e1 n1 # th i7 - ts M92 N2 # m a2 # k O5 - th e3 # z o2 n2 - E6 p6 # d e5 n5 # m o8 t8 # s M7 # f a5 # v 94 # n a2 w2 - d O5 # sp k u4 Nm4 - J M1 # k O5 - th e3 # G 9X1 j1 # t aX6 k6 - N E4 n4 # ts OX1 Nm1 # k a6 k6 # z a1 w1 - z i8 k8 # sp}', '0010485': '{ts OX1 Nm1 # k o1 Nm1 - N ie8 p8 # l wp7 ie7 n7 - k i1 m1 # s i1 - a1 - n uo1 # d M98 k8 # d E1 m1 # N 9X1 m1 # v 95 j5 # k wp7 aX7 N7 # ts M95 # v a2 N2 # b a8 k8 # sp d e3 # t a7 w7 # z a1 # z u1 Nm1 - z i8 k8 # k O5 # ts M95 # s i1 - a1 - n uo1 # v a2 N2 # b a8 k8 # sp}', '0007024': '{f OX1 Nm1 - d o7 # k uo3 # th 9X2 j2 - ts O2 # G E1 n1 - n a1 - r o1 # G a6 t6 - t u1 - s o1 # k u4 Nm4 # x o1 Nm1 # t o2 j2 # sp x i1 # h O7 # b 9X6 t6 # b a7 j7 # ts OX1 Nm1 # t a5 m5 # ts 9X7 n7 # G 9X2 n2 # d 9X1 j1 # sp ts e1 n1 # t 9X6 t6 - k a3 # d 9X5 w5 - ts M92 N2 # v a2 # t wp2 a2 n2 - th aX5 N5 # b a1 # ts 9X7 n7 # G 9X2 n2 # J 9X6 t6 # sp}'}
    # m={'0008618': '{ts wp1 ie1 n1 - z a1 # z a1 w1 - th o1 Nm1 # ts O1 # z aX2 N2 # b e5 n5 - s E1 # ie1 n1 - s 93 # s E4 # z a1 - t aX1 N1 # a6 p6 - l M8 k8 # l e1 n1 - d M92 N2 # v EX2 J2 - d a1 j1 # b a1 # sp k O2 n2 # N M92 j2 - z 9X1 n1 # l O1 - l aX5 N5 # v e2 # a1 n1 - n i1 J1 # ts 9X8 t8 - t M7 # v a2 # u2 n2 - t aX6 k6 # z a1 w1 - th o1 Nm1 # sp}', '0011651': '{h o2 j2 # th a5 N5 # m M92 j2 - h a1 j1 # n aX1 m1 # h a1 j1 - N i2 n2 # x o1 Nm1 - ts aX1 m1 # m M92 j2 - b 9X3 j3 # sp z M95 j5 # s M7 # ie3 m3 - ts 97 # k uo3 # x o1 Nm1 - k wp1 9X1 n1 # N a1 # sp k wp1 9X1 n1 - d o7 j7 # s i1 - r i1 # k u2 Nm2 # l M8 k8 - l M97 N7 # z 9X1 n1 - k wp1 9X1 n1 # d M98 k8 # sp i1 - r a1 n1 # h 9X7 w7 - th wp4 9X4 n4 # sp th M8 k8 - h ie7 n7 # d 98 t8 # t 9X5 n5 - k o1 Nm1 # l 95 n5 # sp J aX2 m2 # z EX2 J2 # l a7 j7 # t i3 J3 # i6 t6 - l i6 p6 - f # sp}', '0000582': '{N M92 j2 # s 9X5 w5 # th 9X8 t8 - s M7 # h aX1 j1 # J M4 N4 # N M92 j2 # x O5 # sp M5 N5 - f O5 # sp d o1 j1 - x i1 # k u4 Nm4 # s E4 # z i8 k8 - ts wp3 ie3 n3 # EX5 J5 # m aX6 t6 # sp x o1 Nm1 # J i2 n2 # v a2 w2 # m aX8 t8 # k uo3 # d o5 j5 - f M91 N1 # sp v i2 # J M4 N4 # N M92 j2 # s 9X5 w5 # x o1 Nm1 # f a3 j3 # l a2 # x o1 Nm1 # k O5 # ts u6 t6 # s 97 - s e8 t8 # n a2 w2 # k a3 # sp}', '0001386': '{s i1 n1 # ts a2 w2 # k wp5 i5 - x EX6 k6 # ts i7 J7 # th i7 # sp b ie3 n3 # sp}', '0010420': '{n e5 w5 # ts i5 J5 - k wp2 ie2 n2 # tS aX1 m1 # th M8 k8 - s M7 # m uo5 n5 # z a3 j3 - k wp6 ie6 t6 # v 9X5 n5 - d e2 # th 9X1 m1 - h u8 t8 # th M91 N1 - m a7 j7 # v 95 j5 # ts u1 Nm1 - k uo6 k6 # sp h aX4 j4 # d 9X2 w2 - t M1 # N aX1 j1 # t M2 # b 9X1 j1 - z 92 # sp}'}
    # m = {"test": "{d 9X1 j1 # l a2 # d wp7 a7 n7 # v aX1 n1 - b a3 n3 # v i5 - z u7 # sp ts O1 # t ie5 N5 # v ie8 t8 #  sp v i1 - b i1 # k O5 # h a1 j1 # v aX1 n1 - f OX2 Nm2 # 93 # h a6 t6 # n 92 # v a2 # th EX2 J2 - f o5 - h o2 - ts i5 - m i1 J1 #  sp}"}
    # d_controls = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0] 
    # p_controls = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.0] 
    # e_controls = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    # d_controls = torch.tensor([d_controls])
    # p_controls = torch.tensor([p_controls])
    # e_controls = torch.tensor([e_controls])

    for file,text in m.items():
        texts = [text]
        sequences = [get_sequence_from_text(text) for text in texts]

        # create batch with size 1
        ids = raw_texts = ["validation"]*len(texts)
        speakers = np.array([0]*len(texts))
        texts = np.array([np.array(seq) for seq in sequences])
        texts = pad_1D(texts)
        text_lens = np.array([len(seq) for seq in sequences])
        batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
        batch = to_device(batch, device)

        with torch.no_grad():
            # predict mel spectrogram
            output = model(
                *(batch[2:]),
                # p_control=p_controls,
                # e_control=e_controls,
                # d_control=d_controls
            )
                
            mel_len = output[9][0].item()
            mel_prediction = output[1][0, :mel_len].detach().transpose(0, 1)

            # convert mel to waveform using hifigan
            hf_audio, hf_samplerate = hifi_vocoder.inference(mel_prediction.reshape(1,mel_prediction.shape[0], mel_prediction.shape[1]).to(device), device)
            hf_audio_denoised = hifi_denoiser(hf_audio.reshape([1,len(hf_audio)]), strength=0.01)[:, 0].squeeze().cpu().numpy()
            # os.system(f"mkdir compare/{file}/")
            write(f"compare/{file}-Fastspeech.wav", 22050, hf_audio_denoised.astype(np.int16))

        # import IPython.display as ipd
        # ipd.Audio(hf_audio_denoised, rate=22050)
        