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

# def main(args):
#     # Initialize Weights & Biases
#     wandb.init(project='your_project_name', entity='your_wandb_entity')

#     args = {
#         'optimization': {
#             'lr': 0.001,
#             'epochs': 30,
#         },
#         'data': {
#             'batch_size': 64,
#             'pin_mem': True,
#             'num_workers': 4,
#             'root_path': './data',
#         }
#     }

#     # Set up device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load the pre-trained model from wandb
#     encoder, predictor = load_model_from_wandb('your_wandb_entity', 'your_project_name', 'ijepa_best_model', device)
#     encoder = encoder.to(device)
#     predictor = predictor.to(device)

#     # Make data transforms
#     transform = make_transforms(crop_size=32, crop_scale=(0.8, 1.0), gaussian_blur=False,
#                                 horizontal_flip=True, color_distortion=True, color_jitter=0.4)

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
import copy
import logging
import sys
import yaml
import wandb
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset, DataLoader

from src.datasets.imagenet1k import make_cifar_tr_val
from src.transforms import make_transforms
from src.utils.logging import AverageMeter
from src.utils.distributed import init_distributed
from src.helper import load_checkpoint, init_model, init_opt

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def load_model_from_wandb(entity, project, model_name, device):
    run = wandb.init(project=project, entity=entity)
    artifact = run.use_artifact(model_name + ':latest', type='model')
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, 'latest.pth.tar')

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    encoder, predictor = init_model(device=device)
    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    return encoder, predictor

def get_subset(dataset, percentage):
    dataset_size = len(dataset)
    subset_size = int(dataset_size * percentage / 100)
    indices = list(range(dataset_size))
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)

def validate(val_loader, model, device):
    model.eval()
    val_loss = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images, return_avg_embed=True)
            loss = F.cross_entropy(outputs, targets)
            val_loss.update(loss.item(), images.size(0))

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return val_loss.avg, accuracy

def train_supervised(args, model, train_loader, val_loader, device, epochs, wandb_run):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['optimization']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss_meter = AverageMeter()
        model.train()

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, return_avg_embed=True)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), images.size(0))

        scheduler.step()

        # Validation step
        val_loss, val_accuracy = validate(val_loader, model, device)
        logger.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss_meter.avg:.4f}, '
                    f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Log to wandb
        wandb_run.log({
            'Epoch': epoch + 1,
            'Train Loss': train_loss_meter.avg,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_accuracy,
        })

def main():
    # Initialize Weights & Biases
    wandb.init(project='your_project_name', entity='your_wandb_entity')

    args = {
        'optimization': {
            'lr': 0.001,
            'epochs': 30,
        },
        'data': {
            'batch_size': 32,
            'pin_mem': True,
            'num_workers': 10,
            'root_path': '$replace_this_with_absolute_path_to_your_datasets_directory',
            'crop_size': 224,
            'crop_scale': [0.3, 1.0],
            'use_color_distortion': False,
            'use_gaussian_blur': False,
            'use_horizontal_flip': False,
            'color_jitter_strength': 0.0,
        }
    }

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model from wandb
    encoder, predictor = load_model_from_wandb('your_wandb_entity', 'your_project_name', 'ijepa_best_model', device)
    encoder = encoder.to(device)
    predictor = predictor.to(device)

    # Make data transforms
    transform = make_transforms(crop_size=224, crop_scale=(0.3, 1.0), gaussian_blur=False,
                                horizontal_flip=False, color_distortion=False, color_jitter=0.0)

    # Load CIFAR-10 dataset
    train_dataset, val_dataset, _, val_loader, _, _ = make_cifar_tr_val(
        transform=transform,
        batch_size=args['data']['batch_size'],
        pin_mem=args['data']['pin_mem'],
        num_workers=args['data']['num_workers'],
        root_path=args['data']['root_path'],
        copy_data=True,
        drop_last=True)

    # Define the percentages of data to use
    percentages = [10, 30, 50, 75, 100]

    for percentage in percentages:
        subset = get_subset(train_dataset, percentage)
        train_loader = DataLoader(subset, batch_size=args['data']['batch_size'], shuffle=True,
                                  pin_memory=args['data']['pin_mem'], num_workers=args['data']['num_workers'])

        # Clone the model for each percentage to avoid contamination between runs
        model = copy.deepcopy(encoder)
        model = DistributedDataParallel(model, device_ids=[device])

        # Create a new wandb run for each percentage
        wandb_run = wandb.init(project='your_project_name', entity='your_wandb_entity',
                               name=f'supervised_{percentage}pct')

        # Train the model
        train_supervised(args, model, train_loader, val_loader, device, args['optimization']['epochs'], wandb_run)

        # Finish the wandb run
        wandb_run.finish()

if __name__ == "__main__":
    main()
