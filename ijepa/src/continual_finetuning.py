
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

# from src.masks.multiblock import MaskCollator as MBMaskCollator
# from src.masks.utils import apply_masks
# from src.utils.distributed import (
#     init_distributed,
#     AllReduce
# )
# from src.utils.logging import (
#     CSVLogger,
#     gpu_timer,
#     grad_logger,
#     AverageMeter)
# from src.utils.tensors import repeat_interleave_batch
# from src.datasets.imagenet1k import make_cifar_tr_val

# from src.helper import (
#     load_checkpoint,
#     init_model,
#     init_opt)
# from src.transforms import make_transforms

# # --
# log_timings = True
# log_freq = 1
# checkpoint_freq = 1
# # --

# _GLOBAL_SEED = 0
# np.random.seed(_GLOBAL_SEED)
# torch.manual_seed(_GLOBAL_SEED)
# torch.backends.cudnn.benchmark = True

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger()

# best_val_loss = float('inf')
# best_model_path = None


# # import os
# # import logging
# # import sys
# # import yaml
# # import wandb
# # import copy
# # import numpy as np
# # import torch
# # import torch.multiprocessing as mp
# # import torch.nn.functional as F
# # from torch.nn.parallel import DistributedDataParallel
# # from src.utils.distributed import init_distributed, AllReduce
# # from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
# # from src.datasets.imagenet1k import make_cifar_tr_val
# # from src.helper import load_checkpoint, init_model, init_opt
# # from src.transforms import make_transforms
# # from src.masks.multiblock import MaskCollator as MBMaskCollator
# # --
# log_timings = True
# log_freq = 1
# checkpoint_freq = 1
# # --

# _GLOBAL_SEED = 0
# np.random.seed(_GLOBAL_SEED)
# torch.manual_seed(_GLOBAL_SEED)
# torch.backends.cudnn.benchmark = True

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger()

# best_val_loss = float('inf')
# best_model_path = None

# def main(args, resume_preempt=False):
#     world_size, rank = init_distributed()
#     for data_fraction in [0.1, 0.3, 0.5, 0.75, 1.0]:

#         global best_val_loss, best_model_path
#         # ----------------------------------------------------------------------- #
#         #  PASSED IN PARAMS FROM CONFIG FILE
#         # ----------------------------------------------------------------------- #

#         # -- META
#         use_bfloat16 = args['meta']['use_bfloat16']
#         model_name = args['meta']['model_name']
#         load_model = args['meta']['load_checkpoint'] or resume_preempt
#         r_file = args['meta']['read_checkpoint']
#         copy_data = args['meta']['copy_data']
#         if not torch.cuda.is_available():
#             device = torch.device('cpu')
#         else:
#             device = torch.device('cuda:0')
#             torch.cuda.set_device(device)

#         # -- DATA
#         use_gaussian_blur = args['data']['use_gaussian_blur']
#         use_horizontal_flip = args['data']['use_horizontal_flip']
#         use_color_distortion = args['data']['use_color_distortion']
#         color_jitter = args['data']['color_jitter_strength']
#         # --
#         batch_size = args['data']['batch_size']
#         pin_mem = args['data']['pin_mem']
#         num_workers = args['data']['num_workers']
#         root_path = args['data']['root_path']
#         image_folder = args['data']['image_folder']
#         crop_size = args['data']['crop_size']
#         crop_scale = args['data']['crop_scale']
#         # --

#         # -- MASK
#         allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
#         patch_size = args['mask']['patch_size']  # patch-size for model training
#         num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
#         min_keep = args['mask']['min_keep']  # min number of patches in context block
#         enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
#         num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
#         pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
#         aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
#         # --

#         # -- OPTIMIZATION
#         wd = float(args['optimization']['weight_decay'])
#         final_wd = float(args['optimization']['final_weight_decay'])
#         num_epochs = args['optimization']['epochs']
#         warmup = args['optimization']['warmup']
#         start_lr = args['optimization']['start_lr']
#         lr = args['optimization']['lr']
#         final_lr = args['optimization']['final_lr']
#         accum_steps = 64 # simulate 2048 batch size
#         # -- LOGGING
#         folder = args['logging']['folder']
#         tag = args['logging']['write_tag']

#         dump = os.path.join(folder, 'params-supervised.yaml')
#         with open(dump, 'w') as f:
#             yaml.dump(args, f)
#         # ----------------------------------------------------------------------- #

#         try:
#             mp.set_start_method('spawn')
#         except Exception:
#             pass

#         # -- init torch distributed backend
#         world_size, rank = init_distributed()
#         logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
#         if rank > 0:
#             logger.setLevel(logging.ERROR)

#         # -- log/checkpointing paths
#         log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
#         save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
#         latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
#         load_path = None
#         if load_model:
#             load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

#         # -- make csv_logger
#         csv_logger = CSVLogger(log_file,
#                             ('%d', 'epoch'),
#                             ('%d', 'itr'),
#                             ('%.5f', 'loss'),
#                             ('%d', 'time (ms)'))

#         # -- init model
#         encoder, predictor = init_model(
#             device=device,
#             patch_size=args['mask']['patch_size'],
#             crop_size=crop_size,
#             pred_depth=args['meta']['pred_depth'],
#             pred_emb_dim=args['meta']['pred_emb_dim'],
#             model_name=model_name)
#         target_encoder = copy.deepcopy(encoder)
        
#         # -- make data transforms
#         transform = make_transforms(
#             crop_size=crop_size,
#             crop_scale=crop_scale,
#             gaussian_blur=use_gaussian_blur,
#             horizontal_flip=use_horizontal_flip,
#             color_distortion=use_color_distortion,
#             color_jitter=color_jitter)

#         # -- init data-loaders/samplers
#         # train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler = make_cifar_tr_val(
#         #     transform=transform,
#         #     batch_size=batch_size,
#         #     collator=None,
#         #     pin_mem=pin_mem,
#         #     num_workers=num_workers,
#         #     world_size=world_size,
#         #     rank=rank,
#         #     root_path=root_path,
#         #     copy_data=copy_data,
#         #     drop_last=True,
#         # )

#         # -- make data transforms
#         mask_collator = MBMaskCollator(
#             input_size=crop_size,
#             patch_size=patch_size,
#             pred_mask_scale=pred_mask_scale,
#             enc_mask_scale=enc_mask_scale,
#             aspect_ratio=aspect_ratio,
#             nenc=num_enc_masks,
#             npred=num_pred_masks,
#             allow_overlap=allow_overlap,
#             min_keep=min_keep)
        
#         train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler = make_cifar_tr_val(
#             transform=transform,
#             collator=mask_collator,
#             batch_size=batch_size,
#             pin_mem=pin_mem,
#             num_workers=num_workers,
#             world_size=world_size,
#             rank=rank,
#             root_path=root_path,
#             copy_data=copy_data,
#             drop_last=True,
#             data_fraction=data_fraction
#         )

#         ipe = len(train_loader)

#         # -- init optimizer and scheduler
#         optimizer, scaler, scheduler, wd_scheduler = init_opt(
#             encoder=target_encoder,
#             predictor=predictor,
#             wd=wd,
#             final_wd=final_wd,
#             start_lr=start_lr,
#             ref_lr=lr,
#             final_lr=final_lr,
#             iterations_per_epoch=ipe,
#             warmup=warmup,
#             num_epochs=num_epochs,
#             ipe_scale=1.0,
#             use_bfloat16=use_bfloat16)
        


#         start_epoch = 0

#         # -- load training checkpoint
#         if load_model:
#             target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
#                 device=device,
#                 r_path=load_path,
#                 encoder=target_encoder,
#                 predictor=None,
#                 target_encoder=None,
#                 opt=optimizer,
#                 scaler=scaler)
#             for _ in range(start_epoch*ipe):
#                 scheduler.step()
#                 wd_scheduler.step()

#         def save_checkpoint(epoch):
#             save_dict = {
#                 'encoder': target_encoder.state_dict(),
#                 'opt': optimizer.state_dict(),
#                 'scaler': None if scaler is None else scaler.state_dict(),
#                 'epoch': epoch,
#                 'loss': loss_meter.avg,
#                 'batch_size': batch_size,
#                 'world_size': world_size,
#                 'lr': lr
#             }
#             if rank == 0:
#                 torch.save(save_dict, latest_path)
#                 # if (epoch + 1) % checkpoint_freq == 0:
#                 #     torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
#             return latest_path

#         def validate(val_loader, target_encoder, device='cuda:0'):
#             target_encoder.eval()
#             val_loss = AverageMeter()
#             correct = 0
#             total = 0

#             with torch.no_grad():
#                 # for itr, (udata, labels) in enumerate(val_loader):
#                 for itr, (udata, masks_enc, masks_pred) in enumerate(val_loader):
#                     imgs = udata[0].to(device, non_blocking=True)
#                     labels = udata[1].to(device, non_blocking=True)

#                     with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
#                         outputs = target_encoder(imgs, return_avg_embed=True)
#                         loss = F.cross_entropy(outputs, labels)
                    
#                     val_loss.update(loss.item(), imgs.size(0))
                    
#                     _, predicted = torch.max(outputs, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()

#             accuracy = 100 * correct / total
#             return val_loss.avg, accuracy

#         run = wandb.init(project='ijepa_tr', entity='byyoung3', name=f'{tag}_continual_ft_fraction_{data_fraction}')

#         artifact = run.use_artifact('byyoung3/model-registry/jepa_base:v1', type='model')
#         artifact_dir = artifact.download()

#         # Load target encoder from checkpoint
#         checkpoint = torch.load(os.path.join(artifact_dir, 'jepa-latest.pth.tar'), map_location='cuda:0')

#         # Remove the 'module.' prefix from the keys
#         state_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}

#         print(state_dict.keys())  # Print the keys to see what is available in the checkpoint
#         target_encoder = DistributedDataParallel(target_encoder, static_graph=True)
#         target_encoder.load_state_dict(state_dict)
        
#         target_encoder = DistributedDataParallel(target_encoder, static_graph=True)

        
#         for epoch in range(start_epoch, num_epochs):
#             logger.info('Epoch %d' % (epoch + 1))

#             # -- update distributed-data-loader epoch
#             train_sampler.set_epoch(epoch)

#             loss_meter = AverageMeter()
#             time_meter = AverageMeter()

#             target_encoder.train()

#             optimizer.zero_grad()
#             # for itr, (udata, labels) in enumerate(train_loader):
#             for itr, (udata, masks_enc, masks_pred) in enumerate(train_loader):
#                 imgs = udata[0].to(device, non_blocking=True)
#                 labels = udata[1].to(device, non_blocking=True)

#                 def train_step():
#                     _new_lr = scheduler.step()
#                     _new_wd = wd_scheduler.step()

#                     outputs = target_encoder(imgs, return_avg_embed=True)
#                     loss = F.cross_entropy(outputs, labels)
#                     loss = AllReduce.apply(loss)

#                     # Step 1. Backward & step
#                     if use_bfloat16:
#                         scaler.scale(loss).backward()
#                     else:
#                         loss.backward()

#                     # Accumulate gradients
#                     if (itr + 1) % accum_steps == 0:
#                         if use_bfloat16:
#                             scaler.step(optimizer)
#                             scaler.update()
#                         else:
#                             optimizer.step()
#                         optimizer.zero_grad()

#                     return (float(loss), _new_lr, _new_wd)


#                 (loss, _new_lr, _new_wd), etime = gpu_timer(train_step)

#                 loss_meter.update(loss, imgs.size(0))
#                 time_meter.update(etime, imgs.size(0))

#                 if (itr + 1) % log_freq == 0:
#                     logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{itr + 1}/{ipe}], '
#                                 f'Loss: {loss_meter.avg:.4f}, Time: {time_meter.avg:.2f}s')

#             # Validation step
#             val_loss, val_accuracy = validate(val_loader, target_encoder)

#             logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

#             # Log to wandb
#             run.log({
#                 'Epoch': epoch + 1,
#                 'Train Loss': loss_meter.avg,
#                 'Validation Loss': val_loss,
#                 'Validation Accuracy': val_accuracy,
#             })

#             # Save the best model based on validation loss
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 pth = save_checkpoint(epoch)
#                 best_model_path = pth

#         # run.save(best_model_path)
#         # run.log_model(path=best_model_path, name=f"supervised_best_model_fraction_{data_fraction}", aliases=["best"])
#         run.finish()

# if __name__ == "__main__":
#     main()





import os
import logging
import sys
import yaml
import wandb
import copy
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.datasets.imagenet1k import make_cifar_tr_val
from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms
from src.masks.multiblock import MaskCollator as MBMaskCollator
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

    for data_fraction in [ 1.0]:

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


        print("*"*50)
        print(f"Model: {model_name}")
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
        # train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler = make_cifar_tr_val(
        #     transform=transform,
        #     batch_size=batch_size,
        #     collator=None,
        #     pin_mem=pin_mem,
        #     num_workers=num_workers,
        #     world_size=world_size,
        #     rank=rank,
        #     root_path=root_path,
        #     copy_data=copy_data,
        #     drop_last=True,
        # )

        # -- make data transforms
        mask_collator = MBMaskCollator(
            input_size=crop_size,
            patch_size=patch_size,
            pred_mask_scale=pred_mask_scale,
            enc_mask_scale=enc_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            allow_overlap=allow_overlap,
            min_keep=min_keep)
        
        train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler = make_cifar_tr_val(
            transform=transform,
            collator=mask_collator,
            batch_size=batch_size,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            copy_data=copy_data,
            drop_last=True,
            data_fraction=data_fraction
        )

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
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        # target_encoder = DistributedDataParallel(target_encoder, static_graph=True)

        start_epoch = 0

        # -- load training checkpoint
        if load_model:
            _, _, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
                device=device,
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
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
                # for itr, (udata, labels) in enumerate(val_loader):
                for itr, (udata, masks_enc, masks_pred) in enumerate(val_loader):
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

        run = wandb.init(project='ijepa_tr', entity='byyoung3', name=f'{tag}_continual_ft_fraction_{data_fraction}')

        # # artifact = run.use_artifact('byyoung3/model-registry/jepa_base:v1', type='model')
        # # artifact_dir = artifact.download()
        # # load target encoder from checkpoint here 
        # artifact = run.use_artifact('byyoung3/model-registry/jepa_base:v1', type='model')
        # artifact_dir = artifact.download()

        # # Load target encoder from checkpoint
        # checkpoint = torch.load(os.path.join(artifact_dir, 'jepa-latest.pth.tar'), map_location='cuda:0')

        # target_encoder.module.load_state_dict(checkpoint['target_encoder'])

        # # target_encoder.load_checkpoint(checkpoint)
    
        # artifact = run.use_artifact('byyoung3/model-registry/jepa_base:v1', type='model')
        # artifact_dir = artifact.download()

        # # Load target encoder from checkpoint
        # checkpoint = torch.load(os.path.join(artifact_dir, 'jepa-latest.pth.tar'), map_location='cuda:0')

        # # Remove the 'module.' prefix from the keys if necessary
        # state_dict = checkpoint['encoder']
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('module.'):
        #         new_state_dict[k[7:]] = v
        #     else:
        #         new_state_dict[k] = v

        # # Load the modified state dictionary into the target_encoder
        # target_encoder.module.load_state_dict(new_state_dict)
    
        # target_encoder = DistributedDataParallel(target_encoder, static_graph=True)

        # for name, param in target_encoder.named_parameters():
        #     if 'classifier' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        for name, param in target_encoder.named_parameters():
            print(f"Parameter name: {name}, shape: {param.shape}")
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # Define the number of transformer blocks to unfreeze
        # num_layers_to_unfreeze = 3  # Number of transformer blocks to unfreeze at the end
        # total_layers = 12  # Total number of transformer blocks, adjust if different

        # # Determine the range of layers to unfreeze
        # layers_to_unfreeze = range(total_layers - num_layers_to_unfreeze, total_layers)

        # # Print the state dict and modify the requires_grad attribute
        # for name, param in target_encoder.named_parameters():
        #     print(f"Parameter name: {name}, shape: {param.shape}")
        #     if 'classifier' in name:
        #         param.requires_grad = True
        #     else:
        #         # Unfreeze the last few transformer blocks
        #         parts = name.split('.')
        #         if len(parts) > 3 and parts[2].isdigit():
        #             layer_number = int(parts[2])
        #             if layer_number in layers_to_unfreeze:
        #                 param.requires_grad = True
        #             else:
        #                 param.requires_grad = False
        #         else:
        #             param.requires_grad = False       
        num_epochs = 50 
        for epoch in range(start_epoch, num_epochs):
            logger.info('Epoch %d' % (epoch + 1))

            # -- update distributed-data-loader epoch
            train_sampler.set_epoch(epoch)

            loss_meter = AverageMeter()
            time_meter = AverageMeter()

            target_encoder.train()
            print("TRAINING")

            optimizer.zero_grad()
            # for itr, (udata, labels) in enumerate(train_loader):
            for itr, (udata, masks_enc, masks_pred) in enumerate(train_loader):
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

        # run.save(best_model_path)
        # run.log_model(path=best_model_path, name=f"supervised_best_model_fraction_{data_fraction}", aliases=["best"])
        run.finish()

if __name__ == "__main__":
    main()