import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from evaluate import evaluate
from model import CompTransTTSLoss
from utils.model import get_model, get_param_num, get_vocoder
from utils.tools import get_configs_of, log, synth_one_sample, to_device

torch.backends.cudnn.benchmark = True


def train(rank, args, configs, batch_size, num_gpus):
    preprocess_config, model_config, train_config = configs
    if num_gpus > 1:
        init_process_group(
            backend=train_config["dist_config"]['dist_backend'],
            init_method=train_config["dist_config"]['dist_url'],
            world_size=train_config["dist_config"]['world_size'] * num_gpus,
            rank=rank,
        )
    device = torch.device('cuda:{:d}'.format(rank))
    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, model_config, train_config, sort=True, drop_last=True
    )
    error_sample = 0
    for dt in dataset:
        if len(dt['text'])!=len(dt['pitch']):
            print(len(dt['text']),len(dt['pitch']),len(dt['duration']))
            print(dt['id'],dt['speaker'])
            error_sample+=1
    assert error_sample==0
    data_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)
    scaler = amp.GradScaler(enabled=args.use_amp)
    Loss = CompTransTTSLoss(preprocess_config, model_config, train_config).to(device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Training
    if train_config["finetune_single_voice"]["finetune_single_voice"] is True:
        step = train_config["finetune_single_voice"]["pretrained_ckpt_path"]
        step = step.split("/")[-1].replace(".pth.tar", "")
        step = int(step) + 1
    else:
        step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    if rank == 0:
        print("Number of CompTransTTS Parameters: {}\n".format(get_param_num(model)))
        # Init logger
        for p in train_config["path"].values():
            os.makedirs(p, exist_ok=True)
        train_log_path = os.path.join(train_config["path"]["log_path"], "train")
        val_log_path = os.path.join(train_config["path"]["log_path"], "val")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)

        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        outer_bar.n = step - 1 
        outer_bar.update()

    train = True
    while train:
        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batchs in loader:
            if train == False:
                break
            for batch in batchs:
                batch = to_device(batch, device)

                with amp.autocast(args.use_amp):
                    # Forward
                    output = model(*(batch[2:]), step=step)
                    batch[9:11], output = output[-2:], output[:-2] # Update pitch and energy level

                    # Cal Loss
                    losses = Loss(batch, output, step=step)
                    total_loss = losses[0]
                    total_loss = total_loss / grad_acc_step

                # Backward
                scaler.scale(total_loss).backward()

                # Clipping gradients to avoid gradient explosion
                if step % grad_acc_step == 0:
                    scaler.unscale_(optimizer._optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                # Update weights
                optimizer.step_and_update_lr(scaler)
                scaler.update()
                optimizer.zero_grad()

                if rank == 0:
                    if step % log_step == 0:
                        losses = [l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, CTC Loss: {:.4f}, Binarization Loss: {:.4f}, Prosody Loss: {:.4f}".format(
                            *losses
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(train_logger, step, losses=losses)

                    if step % synth_step == 0:
                        fig, fig_attn, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        if fig_attn is not None:
                            log(
                                train_logger,
                                img=fig_attn,
                                tag="Training_attn/step_{}_{}".format(step, tag),
                            )
                        log(
                            train_logger,
                            img=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                        )

                    if step % val_step == 0:
                        model.eval()
                        message = evaluate(device, model, step, configs, val_logger, vocoder, len(losses))
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()

                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    train = False
                    break
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)
        epoch += 1

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    # Set Device
    torch.manual_seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"])
    num_gpus = torch.cuda.device_count()
    batch_size = int(train_config["optimizer"]["batch_size"] / num_gpus)

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(' ---> Automatic Mixed Precision:', args.use_amp)
    print(' ---> Number of used GPU:', num_gpus)
    print(' ---> Batch size per GPU:', batch_size)
    print(' ---> Batch size in total:', batch_size * num_gpus)
    print(" ---> Type of Building Block:", model_config["block_type"])
    print(" ---> Type of Duration Modeling:", "unsupervised" if model_config["duration_modeling"]["learn_alignment"] else "supervised")
    print("=================================================================================================")
    print("Prepare training ...")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, configs, batch_size, num_gpus))
    else:
        train(0, args, configs, batch_size, num_gpus)
