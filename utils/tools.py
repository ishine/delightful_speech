import os
import json
import yaml

import torch
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
import matplotlib
matplotlib.use("Agg")
from scipy.io import wavfile
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from hifigan_2.inference_e2e import HifiGanInferenceE2E
from text import text_to_sequence, symbols, _symbol_to_id


def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def get_variance_level(preprocess_config, model_config, data_loading=True):
    """
    Consider the fact that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
    Outputs:
        pitch_level_tag, energy_level_tag: ["frame", "phone"]
            If data_loading is set True, then it will only be the "frame" for unsupervised duration modeling. 
            Otherwise, it will be aligned with the feature_level in config.
        pitch_feature_level, energy_feature_level: ["frame_level", "phoneme_level"]
            The feature_level in config where the model will learn each variance in this level regardless of the input level.
    """
    learn_alignment = model_config["duration_modeling"]["learn_alignment"] if data_loading else False
    pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
    energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
    assert pitch_feature_level in ["frame_level", "phoneme_level"]
    assert energy_feature_level in ["frame_level", "phoneme_level"]
    pitch_level_tag = "phone" if (not learn_alignment and pitch_feature_level == "phoneme_level") else "frame"
    energy_level_tag = "phone" if (not learn_alignment and energy_feature_level == "phoneme_level") else "frame"
    return pitch_level_tag, energy_level_tag, pitch_feature_level, energy_feature_level


def get_phoneme_level_pitch(duration, pitch):
    # perform linear interpolation
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))

    # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            pitch[i] = np.mean(pitch[pos : pos + d])
        else:
            pitch[i] = 0
        pos += d
    pitch = pitch[: len(duration)]
    return pitch


def get_phoneme_level_energy(duration, energy):
    # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            energy[i] = np.mean(energy[pos : pos + d])
        else:
            energy[i] = 0
        pos += d
    energy = energy[: len(duration)]
    return energy


def to_device(data, device):
    if len(data) == 15:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            languages,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        languages = torch.from_numpy(languages).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        if durations is not None:
            durations = torch.from_numpy(durations).long().to(device)
        if attn_priors is not None:
            attn_priors = torch.from_numpy(attn_priors).float().to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)

        return [
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            languages,
        ]

    if len(data) == 7:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds)


def log(
    logger, step=None, losses=None, fig=None, img=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        logger.add_scalar("Loss/ctc_loss", losses[6], step)
        logger.add_scalar("Loss/bin_loss", losses[7], step)
        logger.add_scalar("Loss/prosody_loss", losses[8], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if img is not None:
        logger.add_image(tag, img, dataformats='HWC')

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].float().detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].float().detach().transpose(0, 1)
    duration = predictions[5][0, :src_len].int().detach().cpu().numpy()

    fig_attn = None
    if learn_alignment:
        attn_prior, attn_soft, attn_hard, attn_hard_dur, attn_logprob = targets[12], *predictions[10]
        attn_prior = attn_prior[0, :src_len, :mel_len].squeeze().detach().cpu().numpy() # text_len x mel_len
        attn_soft = attn_soft[0, 0, :mel_len, :src_len].detach().cpu().transpose(0, 1).numpy() # text_len x mel_len
        attn_hard = attn_hard[0, 0, :mel_len, :src_len].detach().cpu().transpose(0, 1).numpy() # text_len x mel_len
        fig_attn = plot_alignment(
            [
                attn_soft,
                attn_hard,
                attn_prior,
            ],
            ["Soft Attention", "Hard Attention", "Prior"]
        )

    phoneme_prosody_attn = None
    if predictions[11][-1] is not None and model_config["learn_prosody"] and model_config["prosody"]["learn_implicit"]:
        phoneme_prosody_attn = predictions[11][-1][0][:src_len, :mel_len].detach()

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].float().detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].float().detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].float().detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].float().detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats[f"pitch_{pitch_level_tag}"] + stats[f"energy_{energy_level_tag}"][:2] # Should follow the level at data loading time.

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
            phoneme_prosody_attn.cpu().numpy() if phoneme_prosody_attn is not None else None,
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram", "Prosody Alignment"],
        n_attn=1 if phoneme_prosody_attn is not None else 0,
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, fig_attn, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, denoiser, model_config, preprocess_config, path, args):

    multi_speaker = model_config["multi_speaker"]
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    basenames = targets[0]
    durations = []
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].int().detach().cpu().numpy()
        durations.append(predictions[5][i, :src_len].squeeze().cpu().numpy().round().astype('int32'))
        attn_soft = attn_hard = None

        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats[f"pitch_{pitch_level_tag}"] + stats[f"energy_{energy_level_tag}"][:2] # Should follow the level at data loading time.

        fig_save_dir = os.path.join(
            path, str(args.restore_step), "{}_{}.png".format(basename, args.speaker_id)\
                if multi_speaker and args.mode == "single" else "{}.png".format(basename))
        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
            save_dir=fig_save_dir,
        )

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    if denoiser is not None and type(vocoder) is HifiGanInferenceE2E:
        wav_predictions = []
        for mel in mel_predictions:
            mel = mel.unsqueeze(0)
            print("Mel shape", mel.shape)
            wav, _ = vocoder.inference(mel)
            wav_denoised = denoiser(wav.reshape([1, len(wav)]), strength=0.01)[:, 0].squeeze().cpu().numpy()
            print("Wav shape", wav.reshape([1, len(wav)]).shape)
            print("Wav denoised shape", wav_denoised.shape)
            wav_predictions.append(wav_denoised)
    else:
        wav_predictions = vocoder_infer(
            mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
        )        
    denoise = "denoised" if denoiser is not None else ""
    post_process = "post_process" if args.post_process else ""
    
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename, phoneme_seqs, duration in zip(wav_predictions, basenames, targets[3], durations):
        if args.post_process:
            phoneme_seqs = phoneme_seqs.squeeze()
            print(duration.shape)
            
            new_wav = np.array([])
            start = 0
            hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
            for idx, symb_id in enumerate(phoneme_seqs[0:]):
                d_frames = duration[0+idx]*hop_length
                if symb_id == _symbol_to_id["sp"]:
                    new_wav = np.hstack((new_wav, np.zeros(15*hop_length).astype('int32')))
                else:
                    new_wav = np.hstack((new_wav, wav[start:start+d_frames]))
                start += d_frames
                wavfile.write(os.path.join(
                    path, str(args.restore_step), "{}_{}_{}_{}.wav".format(basename, args.speaker_id, denoise, post_process)\
                        if multi_speaker and args.mode == "single" else "{}_{}_{}.wav".format(basename, denoise, post_process)),
                    sampling_rate, new_wav.astype('int16')) # fix here :))
        else:
            wavfile.write(os.path.join(
                path, str(args.restore_step), "{}_{}_{}_{}.wav".format(basename, args.speaker_id, denoise, post_process)\
                    if multi_speaker and args.mode == "single" else "{}_{}_{}.wav".format(basename, denoise, post_process)),
                sampling_rate, wav.astype('int16')) # fix here :))


def plot_mel(data, stats, titles, n_attn=0, save_dir=None):
    # print([type(dt) for dt in data])
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    if n_attn > 0:
        # Plot Mel Spectrogram
        plot_mel_(fig, axes, data[:-n_attn], stats, titles)

        # Plot Alignment
        xlim = data[0][0].shape[1]
        for i in range(-n_attn, 0):
            im = axes[i][0].imshow(data[i], origin='lower', aspect='auto')
            axes[i][0].set_xlabel('Decoder timestep')
            axes[i][0].set_ylabel('Encoder timestep')
            axes[i][0].set_xlim(0, xlim)
            axes[i][0].set_title(titles[i], fontsize="medium")
            axes[i][0].tick_params(labelsize="x-small")
            axes[i][0].set_anchor("W")
            fig.colorbar(im, ax=axes[i][0])
    else:
        # Plot Mel Spectrogram
        plot_mel_(fig, axes, data, stats, titles)

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()
    return data


def plot_mel_(fig, axes, data, stats, titles, tight_layout=True):
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean
    if tight_layout:
        fig.tight_layout()

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax
    for i in range(len(data)):
        if not data[i]:
            continue
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato", linewidth=.7)
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet", linewidth=.7)
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )


# def plot_single_alignment(alignment, info=None, save_dir=None):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
#     fig.colorbar(im, ax=ax)
#     xlabel = 'Decoder timestep'
#     if info is not None:
#         xlabel += '\n\n' + info
#     plt.xlabel(xlabel)
#     plt.ylabel('Encoder timestep')
#     plt.tight_layout()

#     fig.canvas.draw()
#     data = save_figure_to_numpy(fig)
#     if save_dir is not None:
#         plt.savefig(save_dir)
#     plt.close()
#     return data


def plot_alignment(data, titles=None, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, figsize=[6,4],dpi=300)
    plt.subplots_adjust(top = 0.9, bottom = 0.1, right = 0.95, left = 0.05)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        im = data[i]
        axes[i].imshow(im, origin='lower')
        axes[i].set_xlabel('Audio timestep')
        axes[i].set_ylabel('Text timestep')
        axes[i].set_ylim(0, im.shape[0])
        axes[i].set_xlim(0, im.shape[1])
        axes[i].set_title(titles[i], fontsize='medium')
        axes[i].tick_params(labelsize='x-small') 
        axes[i].set_anchor('W')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()
    return data


def plot_embedding(out_dir, embedding, embedding_speaker_id, gender_dict, filename='embedding.png'):
    colors = 'r','b'
    labels = 'Female','Male'

    data_x = embedding
    data_y = np.array([gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10,10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_3D(inputs, B, T, L):
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, :np.shape(input_)[0], :np.shape(input_)[1]] = input_
    return inputs_padded


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
