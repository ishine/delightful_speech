import sys
from matplotlib import pyplot as plt
sys.path.append('..')

import torch
import yaml
import numpy as np
from scipy.io.wavfile import write


if __name__ == "__main__":
    # 0 - ids,  1 - raw_texts, 2 - speakers, 3 - texts,
    # 4 - src_lens, 5 - max_src_len, 6 - mels, 7 - mel_lens,
    # 8 - max_mel_len, 9 - pitches, 10 - energies, 11 -durations

    # convert symbols to vector
    # texts = ["{k a6 k6 # b ie5 n5 - ts M5 N5 # s a1 # k uo3 # v ie1 m1 # v e1 # a1 # J M1 # b ie5 n5 - ts M5 N5 # d M92 N2 # t ie1 w1 - h wp5 a5 # sp J M1 # G 9X1 j1 # ie3 - ts aX3 j3 # sp}"]
    # ref_mel = np.load("preprocessed_data/NgocHuyenNews_fmax-8000/mel/NgocHuyen-mel-0006268.npy")
    # print(ref_mel.shape)
    # get fastspeech model
    import os 
    import random
    random.seed(1)
    os.system('mkdir -p analysis/')
    valid_path = "preprocessed_data/NgocHuyenNews_fmax-8000/checked_valid.txt"

    with open(valid_path, "r") as f:
        data = f.read().split('\n')[:-1]
    m = {}
    for i in range(5):
        tmp = random.choice(data)
        file,_,text,_ = tmp.split('|')
        m[file]=text
        os.system(f'mkdir -p analysis/{file}')
        print(file,text)
        ref_mel = np.load(f"{mel_path}/NgocHuyen-mel-{file}.npy")
        generate_wav([text],ref_mel,f"analysis/{file}/origin_ref.wav")
        os.system(f'cp {wav_path}/{file}.wav analysis/{file}/origin_wav.wav')
        for i in range(3):
            tmp = random.choice(data)
            rel_mel_file = tmp.split('|')[0]
            ref_mel = np.load(f"{mel_path}/NgocHuyen-mel-{rel_mel_file}.npy")
            generate_wav([text],ref_mel,f"analysis/{file}/{rel_mel_file}-ref.wav")
            
    # create batch with size 1
print("-----------")
print(m)