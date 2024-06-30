# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
import wandb 
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.datasets.imagenet1k import make_cifar_tr_val



from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch


from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms



# --
log_timings = True
log_freq = 1
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)


    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --


    # -- OPTIMIZATION
    num_epochs = args['optimization']['epochs']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    warmup = args['optimization']['warmup']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-supervised.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%d', 'time (ms)'))

    # -- init model
    # model = init_model(device=device, model_name=model_name)
    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    
        
    if torch.cuda.device_count() > 1:

        model = DistributedDataParallel(model)

    # # -- make data transforms
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(crop_size, scale=(crop_scale, 1.0)),
    #     transforms.RandomHorizontalFlip() if use_horizontal_flip else transforms.Lambda(lambda x: x),
    #     transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.1) if use_color_distortion else transforms.Lambda(lambda x: x),
    #     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)) if use_gaussian_blur else transforms.Lambda(lambda x: x),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)    

    # -- init data-loaders/samplers
    # _, supervised_loader, supervised_sampler = make_cifar10(
    #     transform=transform,
    #     batch_size=batch_size,
    #     pin_mem=pin_mem,
    #     training=True,
    #     num_workers=num_workers,
    #     world_size=world_size,
    #     rank=rank,
    #     root_path=root_path,
    #     copy_data=copy_data,
    #     drop_last=True)

    train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler = make_cifar_tr_val(
        transform=transform,
        batch_size=batch_size,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        copy_data=copy_data,
        drop_last=True)


    ipe = len(train_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        predictor=predictor,
        encoder=encoder,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        model, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            model=model,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

#     # -- TRAINING LOOP
#     for epoch in range(start_epoch, num_epochs):
#         logger.info('Epoch %d' % (epoch + 1))

#         # -- update distributed-data-loader epoch
#         supervised_sampler.set_epoch(epoch)

#         loss_meter = AverageMeter()
#         time_meter = AverageMeter()

#         for itr, (imgs, targets) in enumerate(supervised_loader):

#             imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

#             def train_step():
#                 _new_lr = scheduler.step()
#                 optimizer.zero_grad()
#                 with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
#                     outputs = encoder(imgs, return_avg_embed=True)
#                     loss = F.cross_entropy(outputs, targets)
#                     print(loss)
#                 if use_bfloat16:
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     optimizer.step()

#                 loss = AllReduce.apply(loss)
#                 return float(loss), _new_lr
            
#             train_step()

# #             loss, _new_lr, etime = gpu_timer(train_step)
# #             loss_meter.update(loss)
# #             time_meter.update(etime)

# #             # -- Logging
# #             def log_stats():
# #                 csv_logger.log(epoch + 1, itr, loss, time_meter.val)
# #                 if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
# #                     logger.info('[%d, %5d] loss: %.3f [lr: %.2e] [mem: %.2e] (%.1f ms)'
# #                                 % (epoch + 1, itr, loss_meter.avg, _new_lr,
# #                                    torch.cuda.max_memory_allocated() / 1024.**2, time_meter.avg))

# #             log_stats()
# #             assert not np.isnan(loss), 'loss is nan'

# #         # -- Save Checkpoint after every epoch
# #         logger.info('avg. loss %.3f'

# #  % loss_meter.avg)
# #         save_checkpoint(epoch + 1)


# if __name__ == "__main__":
#     main()

    # Initialize wandb
    wandb.init(project='cifar10_training', config=args)
    
    
    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        encoder.train()
        for itr, (imgs, targets) in enumerate(train_loader):

            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            def train_step():
                _new_lr = scheduler.step()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    outputs = encoder(imgs, return_avg_embed=True)
                    loss = F.cross_entropy(outputs, targets)
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                loss = AllReduce.apply(loss)
                return float(loss), _new_lr
            
            train_loss, _new_lr = train_step()

            loss_meter.update(train_loss)
            # Log to wandb
            wandb.log({"Train Loss": train_loss, "Learning Rate": _new_lr, "Epoch": epoch})

        # -- Validation Loop
        encoder.eval()
        val_loss_meter = AverageMeter()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = encoder(imgs, return_avg_embed=True)
                val_loss = F.cross_entropy(outputs, targets)
                val_loss_meter.update(val_loss.item())

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_accuracy = 100 * correct / total
        logger.info(f'Validation Loss: {val_loss_meter.avg:.3f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Log validation metrics to wandb
        wandb.log({"Validation Loss": val_loss_meter.avg, "Validation Accuracy": val_accuracy, "Epoch": epoch})

        # -- Save Checkpoint after every epoch
        # save_checkpoint(epoch + 1)

if __name__ == "__main__":
    main()





