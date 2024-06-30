# # import os
# # import copy
# # import logging
# # import sys
# # import yaml
# # import wandb
# # import numpy as np

# # import torch
# # import torch.multiprocessing as mp
# # import torch.nn.functional as F
# # from torch.nn.parallel import DistributedDataParallel
# # from torch.utils.data import Subset, DataLoader

# # from src.datasets.imagenet1k import make_cifar_tr_val
# # from src.transforms import make_transforms
# # from src.utils.logging import AverageMeter
# # from src.utils.distributed import init_distributed
# # from src.helper import load_checkpoint, init_model, init_opt

# # # Set up logging
# # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# # logger = logging.getLogger()

# # def load_model_from_wandb(entity, project, model_name, device):
# #     run = wandb.init(project=project, entity=entity)
# #     artifact = run.use_artifact(model_name + ':latest', type='model')
# #     artifact_dir = artifact.download()
# #     model_path = os.path.join(artifact_dir, 'latest.pth.tar')

# #     # Load model checkpoint
# #     checkpoint = torch.load(model_path, map_location=device)
# #     encoder, predictor = init_model(device=device)
# #     encoder.load_state_dict(checkpoint['encoder'])
# #     predictor.load_state_dict(checkpoint['predictor'])
# #     return encoder, predictor

# # def get_subset(dataset, percentage):
# #     dataset_size = len(dataset)
# #     subset_size = int(dataset_size * percentage / 100)
# #     indices = list(range(dataset_size))
# #     subset_indices = indices[:subset_size]
# #     return Subset(dataset, subset_indices)

# # def validate(val_loader, model, device):
# #     model.eval()
# #     val_loss = AverageMeter()
# #     correct = 0
# #     total = 0

# #     with torch.no_grad():
# #         for images, targets in val_loader:
# #             images = images.to(device)
# #             targets = targets.to(device)
# #             outputs = model(images, return_avg_embed=True)
# #             loss = F.cross_entropy(outputs, targets)
# #             val_loss.update(loss.item(), images.size(0))

# #             _, predicted = torch.max(outputs, 1)
# #             total += targets.size(0)
# #             correct += (predicted == targets).sum().item()

# #     accuracy = 100 * correct / total
# #     return val_loss.avg, accuracy

# # def train_supervised(args, model, train_loader, val_loader, device, epochs, wandb_run):
# #     model.train()
# #     optimizer = torch.optim.Adam(model.parameters(), lr=args['optimization']['lr'])
# #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# #     criterion = torch.nn.CrossEntropyLoss()

# #     for epoch in range(epochs):
# #         train_loss_meter = AverageMeter()
# #         model.train()

# #         for images, targets in train_loader:
# #             images = images.to(device)
# #             targets = targets.to(device)

# #             optimizer.zero_grad()
# #             outputs = model(images, return_avg_embed=True)
# #             loss = criterion(outputs, targets)
# #             loss.backward()
# #             optimizer.step()

# #             train_loss_meter.update(loss.item(), images.size(0))

# #         scheduler.step()

# #         # Validation step
# #         val_loss, val_accuracy = validate(val_loader, model, device)
# #         logger.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss_meter.avg:.4f}, '
# #                     f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# #         # Log to wandb
# #         wandb_run.log({
# #             'Epoch': epoch + 1,
# #             'Train Loss': train_loss_meter.avg,
# #             'Validation Loss': val_loss,
# #             'Validation Accuracy': val_accuracy,
# #         })

# # def main(args):
# #     # Initialize Weights & Biases
# #     wandb.init(project='your_project_name', entity='your_wandb_entity')

# #     args = {
# #         'optimization': {
# #             'lr': 0.001,
# #             'epochs': 30,
# #         },
# #         'data': {
# #             'batch_size': 64,
# #             'pin_mem': True,
# #             'num_workers': 4,
# #             'root_path': './data',
# #         }
# #     }

# #     # Set up device
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #     # Load the pre-trained model from wandb
# #     encoder, predictor = load_model_from_wandb('your_wandb_entity', 'your_project_name', 'ijepa_best_model', device)
# #     encoder = encoder.to(device)
# #     predictor = predictor.to(device)

# #     # Make data transforms
# #     transform = make_transforms(crop_size=32, crop_scale=(0.8, 1.0), gaussian_blur=False,
# #                                 horizontal_flip=True, color_distortion=True, color_jitter=0.4)

# #     # Load CIFAR-10 dataset
# #     train_dataset, val_dataset, _, val_loader, _, _ = make_cifar_tr_val(
# #         transform=transform,
# #         batch_size=args['data']['batch_size'],
# #         pin_mem=args['data']['pin_mem'],
# #         num_workers=args['data']['num_workers'],
# #         root_path=args['data']['root_path'],
# #         copy_data=True,
# #         drop_last=True)

# #     # Define the percentages of data to use
# #     percentages = [10, 30, 50, 75, 100]

# #     for percentage in percentages:
# #         subset = get_subset(train_dataset, percentage)
# #         train_loader = DataLoader(subset, batch_size=args['data']['batch_size'], shuffle=True,
# #                                   pin_memory=args['data']['pin_mem'], num_workers=args['data']['num_workers'])

# #         # Clone the model for each percentage to avoid contamination between runs
# #         model = copy.deepcopy(encoder)
# #         model = DistributedDataParallel(model, device_ids=[device])

# #         # Create a new wandb run for each percentage
# #         wandb_run = wandb.init(project='your_project_name', entity='your_wandb_entity',
# #                                name=f'supervised_{percentage}pct')

# #         # Train the model
# #         train_supervised(args, model, train_loader, val_loader, device, args['optimization']['epochs'], wandb_run)

# #         # Finish the wandb run
# #         wandb_run.finish()

# # if __name__ == "__main__":
# #     main()


# import os
# import copy
# import logging
# import sys
# import yaml
# import wandb
# import numpy as np

# import torch
# import torch.multiprocessing as mp
# import torch.nn.functional as F
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data import Subset, DataLoader

# from src.datasets.imagenet1k import make_cifar_tr_val
# from src.transforms import make_transforms
# from src.utils.logging import AverageMeter
# from src.utils.distributed import init_distributed
# from src.helper import load_checkpoint, init_model, init_opt

# # Set up logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger()

# def load_model_from_wandb(entity, project, model_name, device):
#     run = wandb.init(project=project, entity=entity)
#     artifact = run.use_artifact(model_name + ':latest', type='model')
#     artifact_dir = artifact.download()
#     model_path = os.path.join(artifact_dir, 'latest.pth.tar')

#     # Load model checkpoint
#     checkpoint = torch.load(model_path, map_location=device)
#     encoder, predictor = init_model(device=device)
#     encoder.load_state_dict(checkpoint['encoder'])
#     predictor.load_state_dict(checkpoint['predictor'])
#     return encoder, predictor

# def get_subset(dataset, percentage):
#     dataset_size = len(dataset)
#     subset_size = int(dataset_size * percentage / 100)
#     indices = list(range(dataset_size))
#     subset_indices = indices[:subset_size]
#     return Subset(dataset, subset_indices)

# def validate(val_loader, model, device):
#     model.eval()
#     val_loss = AverageMeter()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, targets in val_loader:
#             images = images.to(device)
#             targets = targets.to(device)
#             outputs = model(images, return_avg_embed=True)
#             loss = F.cross_entropy(outputs, targets)
#             val_loss.update(loss.item(), images.size(0))

#             _, predicted = torch.max(outputs, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()

#     accuracy = 100 * correct / total
#     return val_loss.avg, accuracy

# def train_supervised(args, model, train_loader, val_loader, device, epochs, wandb_run):
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args['optimization']['lr'])
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#     criterion = torch.nn.CrossEntropyLoss()

#     for epoch in range(epochs):
#         train_loss_meter = AverageMeter()
#         model.train()

#         for images, targets in train_loader:
#             images = images.to(device)
#             targets = targets.to(device)

#             optimizer.zero_grad()
#             outputs = model(images, return_avg_embed=True)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             train_loss_meter.update(loss.item(), images.size(0))

#         scheduler.step()

#         # Validation step
#         val_loss, val_accuracy = validate(val_loader, model, device)
#         logger.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss_meter.avg:.4f}, '
#                     f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

#         # Log to wandb
#         wandb_run.log({
#             'Epoch': epoch + 1,
#             'Train Loss': train_loss_meter.avg,
#             'Validation Loss': val_loss,
#             'Validation Accuracy': val_accuracy,
#         })

# def main():
#     # Initialize Weights & Biases
#     wandb.init(project='your_project_name', entity='your_wandb_entity')

#     args = {
#         'optimization': {
#             'lr': 0.001,
#             'epochs': 30,
#         },
#         'data': {
#             'batch_size': 32,
#             'pin_mem': True,
#             'num_workers': 10,
#             'root_path': '$replace_this_with_absolute_path_to_your_datasets_directory',
#             'crop_size': 224,
#             'crop_scale': [0.3, 1.0],
#             'use_color_distortion': False,
#             'use_gaussian_blur': False,
#             'use_horizontal_flip': False,
#             'color_jitter_strength': 0.0,
#         }
#     }

#     # Set up device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load the pre-trained model from wandb
#     encoder, predictor = load_model_from_wandb('your_wandb_entity', 'your_project_name', 'ijepa_best_model', device)
#     encoder = encoder.to(device)
#     predictor = predictor.to(device)

#     # Make data transforms
#     transform = make_transforms(crop_size=224, crop_scale=(0.3, 1.0), gaussian_blur=False,
#                                 horizontal_flip=False, color_distortion=False, color_jitter=0.0)

#     # Load CIFAR-10 dataset
#     train_dataset, val_dataset, _, val_loader, _, _ = make_cifar_tr_val(
#         transform=transform,
#         batch_size=args['data']['batch_size'],
#         pin_mem=args['data']['pin_mem'],
#         num_workers=args['data']['num_workers'],
#         root_path=args['data']['root_path'],
#         copy_data=True,
#         drop_last=True)

#     # Define the percentages of data to use
#     percentages = [10, 30, 50, 75, 100]

#     for percentage in percentages:
#         subset = get_subset(train_dataset, percentage)
#         train_loader = DataLoader(subset, batch_size=args['data']['batch_size'], shuffle=True,
#                                   pin_memory=args['data']['pin_mem'], num_workers=args['data']['num_workers'])

#         # Clone the model for each percentage to avoid contamination between runs
#         model = copy.deepcopy(encoder)
#         model = DistributedDataParallel(model, device_ids=[device])

#         # Create a new wandb run for each percentage
#         wandb_run = wandb.init(project='your_project_name', entity='your_wandb_entity',
#                                name=f'supervised_{percentage}pct')

#         # Train the model
#         train_supervised(args, model, train_loader, val_loader, device, args['optimization']['epochs'], wandb_run)

#         # Finish the wandb run
#         wandb_run.finish()

# if __name__ == "__main__":
#     main()


import os
import logging
import sys
import yaml
import wandb
import os
import copy
import logging

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    AverageMeter)
from src.datasets.imagenet1k import make_cifar_tr_val

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

best_val_loss = float('inf')
best_model_path = None

def main(args, resume_preempt=False):
    global best_val_loss, best_model_path
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    # if not torch.cuda.is_available():
    #     device = torch.device('cpu')
    # else:
    #     device = torch.device('cuda:0')
    #     torch.cuda.set_device(device)
    device = 'cpu'
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

    # -- OPTIMIZATION
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    accum_steps = 64 # simulate 2048 batch size
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
    encoder, predictor = init_model(
        device=device,
        patch_size=args['mask']['patch_size'],
        crop_size=crop_size,
        pred_depth=args['meta']['pred_depth'],
        pred_emb_dim=args['meta']['pred_emb_dim'],
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)
    
    # -- make data transforms
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler = make_cifar_tr_val(
        transform=transform,
        batch_size=batch_size,
        collator=None,
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
        encoder=target_encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=1.0,
        use_bfloat16=use_bfloat16)
    target_encoder = DistributedDataParallel(target_encoder, static_graph=True)

    start_epoch = 0

    # -- load training checkpoint
    if load_model:
        target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=target_encoder,
            predictor=None,
            target_encoder=None,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': target_encoder.state_dict(),
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
            # if (epoch + 1) % checkpoint_freq == 0:
            #     torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
        return latest_path

    def validate(val_loader, target_encoder, device='cuda:0'):
        target_encoder.eval()
        val_loss = AverageMeter()
        correct = 0
        total = 0

        with torch.no_grad():
            for itr, (udata, labels) in enumerate(val_loader):
                imgs = udata[0].to(device, non_blocking=True)
                labels = udata[1].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    outputs = target_encoder(imgs, return_avg_embed=True)
                    loss = F.cross_entropy(outputs, labels)
                
                val_loss.update(loss.item(), imgs.size(0))
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return val_loss.avg, accuracy

    run = wandb.init(project='supervised_ijepa_tr', entity='byyoung3')

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        target_encoder.train()

        optimizer.zero_grad()
        for itr, (udata, labels) in enumerate(train_loader):
            imgs = udata[0].to(device, non_blocking=True)
            labels = udata[1].to(device, non_blocking=True)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                outputs = target_encoder(imgs, return_avg_embed=True)
                loss = F.cross_entropy(outputs, labels)
                loss = AllReduce.apply(loss)

                # Step 1. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Accumulate gradients
                if (itr + 1) % accum_steps == 0:
                    if use_bfloat16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                return (float(loss), _new_lr, _new_wd)
            (loss, _new_lr, _new_wd), etime = gpu_timer(train_step)

            loss_meter.update(loss, imgs.size(0))
            time_meter.update(etime, imgs.size(0))

            if (itr + 1) % log_freq == 0:
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{itr + 1}/{ipe}], '
                            f'Loss: {loss_meter.avg:.4f}, Time: {time_meter.avg:.2f}s')

        # Validation step
        val_loss, val_accuracy = validate(val_loader, target_encoder)

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Log to wandb
        run.log({
            'Epoch': epoch + 1,
            'Train Loss': loss_meter.avg,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_accuracy,
        })

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pth = save_checkpoint(epoch)
            best_model_path = pth

    run.save(best_model_path)
    run.log_model(path=best_model_path, name="supervised_best_model", aliases=["best"])
    run.finish()


if __name__ == "__main__":
    main()
