# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# #

# import os
import subprocess
# import time

# import numpy as np

# from logging import getLogger

# import torch
# import torchvision

# _GLOBAL_SEED = 0
# logger = getLogger()


# def make_imagenet1k(
#     transform,
#     batch_size,
#     collator=None,
#     pin_mem=True,
#     num_workers=8,
#     world_size=1,
#     rank=0,
#     root_path=None,
#     image_folder=None,
#     training=True,
#     copy_data=False,
#     drop_last=True,
#     subset_file=None
# ):
#     dataset = ImageNet(
#         root=root_path,
#         image_folder=image_folder,
#         transform=transform,
#         train=training,
#         copy_data=copy_data,
#         index_targets=False)
#     if subset_file is not None:
#         dataset = ImageNetSubset(dataset, subset_file)
#     logger.info('ImageNet dataset created')
#     dist_sampler = torch.utils.data.distributed.DistributedSampler(
#         dataset=dataset,
#         num_replicas=world_size,
#         rank=rank)
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         collate_fn=collator,
#         sampler=dist_sampler,
#         batch_size=batch_size,
#         drop_last=drop_last,
#         pin_memory=pin_mem,
#         num_workers=num_workers,
#         persistent_workers=False)
#     logger.info('ImageNet unsupervised data loader created')

#     return dataset, data_loader, dist_sampler


# class ImageNet(torchvision.datasets.ImageFolder):

#     def __init__(
#         self,
#         root,
#         image_folder='imagenet_full_size/061417/',
#         tar_file='imagenet_full_size-061417.tar.gz',
#         transform=None,
#         train=True,
#         job_id=None,
#         local_rank=None,
#         copy_data=True,
#         index_targets=False
#     ):
#         """
#         ImageNet

#         Dataset wrapper (can copy data locally to machine)

#         :param root: root network directory for ImageNet data
#         :param image_folder: path to images inside root network directory
#         :param tar_file: zipped image_folder inside root network directory
#         :param train: whether to load train data (or validation)
#         :param job_id: scheduler job-id used to create dir on local machine
#         :param copy_data: whether to copy data from network file locally
#         :param index_targets: whether to index the id of each labeled image
#         """

#         suffix = 'train/' if train else 'val/'
#         data_path = None
#         if copy_data:
#             logger.info('copying data locally')
#             data_path = copy_imgnt_locally(
#                 root=root,
#                 suffix=suffix,
#                 image_folder=image_folder,
#                 tar_file=tar_file,
#                 job_id=job_id,
#                 local_rank=local_rank)
#         if (not copy_data) or (data_path is None):
#             data_path = os.path.join(root, image_folder, suffix)
#         logger.info(f'data-path {data_path}')

#         super(ImageNet, self).__init__(root=data_path, transform=transform)
#         logger.info('Initialized ImageNet')

#         if index_targets:
#             self.targets = []
#             for sample in self.samples:
#                 self.targets.append(sample[1])
#             self.targets = np.array(self.targets)
#             self.samples = np.array(self.samples)

#             mint = None
#             self.target_indices = []
#             for t in range(len(self.classes)):
#                 indices = np.squeeze(np.argwhere(
#                     self.targets == t)).tolist()
#                 self.target_indices.append(indices)
#                 mint = len(indices) if mint is None else min(mint, len(indices))
#                 logger.debug(f'num-labeled target {t} {len(indices)}')
#             logger.info(f'min. labeled indices {mint}')


# class ImageNetSubset(object):

#     def __init__(self, dataset, subset_file):
#         """
#         ImageNetSubset

#         :param dataset: ImageNet dataset object
#         :param subset_file: '.txt' file containing IDs of IN1K images to keep
#         """
#         self.dataset = dataset
#         self.subset_file = subset_file
#         self.filter_dataset_(subset_file)

#     def filter_dataset_(self, subset_file):
#         """ Filter self.dataset to a subset """
#         root = self.dataset.root
#         class_to_idx = self.dataset.class_to_idx
#         # -- update samples to subset of IN1k targets/samples
#         new_samples = []
#         logger.info(f'Using {subset_file}')
#         with open(subset_file, 'r') as rfile:
#             for line in rfile:
#                 class_name = line.split('_')[0]
#                 target = class_to_idx[class_name]
#                 img = line.split('\n')[0]
#                 new_samples.append(
#                     (os.path.join(root, class_name, img), target)
#                 )
#         self.samples = new_samples

#     @property
#     def classes(self):
#         return self.dataset.classes

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         img = self.dataset.loader(path)
#         if self.dataset.transform is not None:
#             img = self.dataset.transform(img)
#         if self.dataset.target_transform is not None:
#             target = self.dataset.target_transform(target)
#         return img, target

# import os
# import time

# import numpy as np
# from logging import getLogger

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

# _GLOBAL_SEED = 0
# np.random.seed(_GLOBAL_SEED)

# # Setting the seed for torch
# torch.manual_seed(_GLOBAL_SEED)
# torch.cuda.manual_seed_all(_GLOBAL_SEED)

# logger = getLogger()


# def make_cifar10(
#     transform,
#     batch_size,
#     collator=None,
#     pin_mem=True,
#     num_workers=8,
#     world_size=1,
#     rank=0,
#     root_path='./data',
#     training=True,
#     copy_data=False,
#     drop_last=True,
#     subset_file=None
# ):
#     dataset = CIFAR10Dataset(
#         root=root_path,
#         transform=transform,
#         train=training,
#         download=True)
#     if subset_file is not None:
#         dataset = CIFAR10Subset(dataset, subset_file)
#     logger.info('CIFAR-10 dataset created')
#     dist_sampler = torch.utils.data.distributed.DistributedSampler(
#         dataset=dataset,
#         num_replicas=world_size,
#         rank=rank)
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         collate_fn=collator,
#         sampler=dist_sampler,
#         batch_size=batch_size,
#         drop_last=drop_last,
#         pin_memory=pin_mem,
#         num_workers=num_workers,
#         persistent_workers=False)
#     logger.info('CIFAR-10 data loader created')

#     return dataset, data_loader, dist_sampler

# def make_cifar_tr_val(
#     transform,
#     batch_size,
#     collator=None,
#     pin_mem=True,
#     num_workers=8,
#     world_size=1,
#     rank=0,
#     root_path='./data',
#     copy_data=False,
#     drop_last=True,
#     subset_file=None
# ):
#     train_dataset = CIFAR10Dataset(
#         root=root_path,
#         transform=transform,
#         train=True,
#         download=True)
    
#     val_dataset = CIFAR10Dataset(
#         root=root_path,
#         transform=transform,
#         train=False,
#         download=True)
    
#     if subset_file is not None:
#         train_dataset = CIFAR10Subset(train_dataset, subset_file)
    
#     logger.info('CIFAR-10 training and validation datasets created')
    
#     train_sampler = torch.utils.data.distributed.DistributedSampler(
#         dataset=train_dataset,
#         num_replicas=world_size,
#         rank=rank)
    
#     val_sampler = torch.utils.data.distributed.DistributedSampler(
#         dataset=val_dataset,
#         num_replicas=world_size,
#         rank=rank)
    
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         collate_fn=collator,
#         sampler=train_sampler,
#         batch_size=batch_size,
#         drop_last=drop_last,
#         pin_memory=pin_mem,
#         num_workers=num_workers,
#         persistent_workers=False)
    
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         collate_fn=collator,
#         sampler=val_sampler,
#         batch_size=batch_size,
#         drop_last=drop_last,
#         pin_memory=pin_mem,
#         num_workers=num_workers,
#         persistent_workers=False)
    
#     logger.info('CIFAR-10 data loaders created for training and validation')

#     return train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler

# class CIFAR10Dataset(torchvision.datasets.CIFAR10):

#     def __init__(
#         self,
#         root,
#         transform=None,
#         train=True,
#         download=True,
#         copy_data=True,
#         index_targets=False
#     ):
#         """
#         CIFAR-10 Dataset

#         :param root: root directory for CIFAR-10 data
#         :param transform: transformations to apply
#         :param train: whether to load train data (or validation)
#         :param download: whether to download the dataset if not present
#         :param copy_data: whether to copy data locally (not applicable for CIFAR-10)
#         :param index_targets: whether to index the id of each labeled image
#         """
#         super(CIFAR10Dataset, self).__init__(root=root, train=train, transform=transform, download=download)
#         logger.info('Initialized CIFAR-10')

#         if index_targets:
#             self.targets = np.array(self.targets)
#             self.samples = [(self.data[i], self.targets[i]) for i in range(len(self.targets))]

#             mint = None
#             self.target_indices = []
#             for t in range(len(self.classes)):
#                 indices = np.squeeze(np.argwhere(self.targets == t)).tolist()
#                 self.target_indices.append(indices)
#                 mint = len(indices) if mint is None else min(mint, len(indices))
#                 logger.debug(f'num-labeled target {t} {len(indices)}')
#             logger.info(f'min. labeled indices {mint}')


# class CIFAR10Subset(object):

#     def __init__(self, dataset, subset_file):
#         """
#         CIFAR-10 Subset

#         :param dataset: CIFAR-10 dataset object
#         :param subset_file: '.txt' file containing indices of CIFAR-10 images to keep
#         """
#         self.dataset = dataset
#         self.subset_file = subset_file
#         self.filter_dataset_(subset_file)

#     def filter_dataset_(self, subset_file):
#         """ Filter self.dataset to a subset """
#         new_samples = []
#         logger.info(f'Using {subset_file}')
#         with open(subset_file, 'r') as rfile:
#             for line in rfile:
#                 idx = int(line.strip())
#                 new_samples.append(self.dataset.samples[idx])
#         self.samples = new_samples

#     @property
#     def classes(self):
#         return self.dataset.classes

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         img, target = self.samples[index]
#         if self.dataset.transform is not None:
#             img = self.dataset.transform(img)
#         if self.dataset.target_transform is not None:
#             target = self.dataset.target_transform(target)
#         return img, target



# def copy_imgnt_locally(
#     root,
#     suffix,
#     image_folder='imagenet_full_size/061417/',
#     tar_file='imagenet_full_size-061417.tar.gz',
#     job_id=None,
#     local_rank=None
# ):
#     if job_id is None:
#         try:
#             job_id = os.environ['SLURM_JOBID']
#         except Exception:
#             logger.info('No job-id, will load directly from network file')
#             return None

#     if local_rank is None:
#         try:
#             local_rank = int(os.environ['SLURM_LOCALID'])
#         except Exception:
#             logger.info('No job-id, will load directly from network file')
#             return None

#     source_file = os.path.join(root, tar_file)
#     target = f'/scratch/slurm_tmpdir/{job_id}/'
#     target_file = os.path.join(target, tar_file)
#     data_path = os.path.join(target, image_folder, suffix)
#     logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

#     tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

#     if not os.path.exists(data_path):
#         if local_rank == 0:
#             commands = [
#                 ['tar', '-xf', source_file, '-C', target]]
#             for cmnd in commands:
#                 start_time = time.time()
#                 logger.info(f'Executing {cmnd}')
#                 subprocess.run(cmnd)
#                 logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
#             with open(tmp_sgnl_file, '+w') as f:
#                 print('Done copying locally.', file=f)
#         else:
#             while not os.path.exists(tmp_sgnl_file):
#                 time.sleep(60)
#                 logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

#     return data_path


import os
import time

import numpy as np
from logging import getLogger

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)

# Setting the seed for torch
torch.manual_seed(_GLOBAL_SEED)
torch.cuda.manual_seed_all(_GLOBAL_SEED)

logger = getLogger()


def make_cifar10(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path='./data',
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    data_fraction=1.0
):
    dataset = CIFAR10Dataset(
        root=root_path,
        transform=transform,
        train=training,
        download=True)
    
    if subset_file is not None:
        dataset = CIFAR10Subset(dataset, subset_file)
    
    if data_fraction < 1.0:
        total_samples = len(dataset)
        subset_size = int(data_fraction * total_samples)
        indices = torch.randperm(total_samples).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:subset_size])
    
    logger.info('CIFAR-10 dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('CIFAR-10 data loader created')

    return dataset, data_loader, dist_sampler

def make_cifar_tr_val(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path='./data',
    copy_data=False,
    drop_last=True,
    subset_file=None,
    data_fraction=1.0
):
    train_dataset = CIFAR10Dataset(
        root=root_path,
        transform=transform,
        train=True,
        download=True)
    
    val_dataset = CIFAR10Dataset(
        root=root_path,
        transform=transform,
        train=False,
        download=True)
    
    if subset_file is not None:
        train_dataset = CIFAR10Subset(train_dataset, subset_file)
    
    if data_fraction < 1.0:
        total_train_samples = len(train_dataset)
        total_val_samples = len(val_dataset)
        train_subset_size = int(data_fraction * total_train_samples)
        val_subset_size = int(data_fraction * total_val_samples)
        train_indices = torch.randperm(total_train_samples).tolist()
        val_indices = torch.randperm(total_val_samples).tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices[:train_subset_size])
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices[:val_subset_size])
    
    logger.info('CIFAR-10 training and validation datasets created')
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=val_dataset,
        num_replicas=world_size,
        rank=rank)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collator,
        sampler=train_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collator,
        sampler=val_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    
    logger.info('CIFAR-10 data loaders created for training and validation')

    return train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler

class CIFAR10Dataset(torchvision.datasets.CIFAR10):

    def __init__(
        self,
        root,
        transform=None,
        train=True,
        download=True,
        copy_data=True,
        index_targets=False
    ):
        """
        CIFAR-10 Dataset

        :param root: root directory for CIFAR-10 data
        :param transform: transformations to apply
        :param train: whether to load train data (or validation)
        :param download: whether to download the dataset if not present
        :param copy_data: whether to copy data locally (not applicable for CIFAR-10)
        :param index_targets: whether to index the id of each labeled image
        """
        super(CIFAR10Dataset, self).__init__(root=root, train=train, transform=transform, download=download)
        logger.info('Initialized CIFAR-10')

        if index_targets:
            self.targets = np.array(self.targets)
            self.samples = [(self.data[i], self.targets[i]) for i in range(len(self.targets))]

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}')


class CIFAR10Subset(object):

    def __init__(self, dataset, subset_file):
        """
        CIFAR-10 Subset

        :param dataset: CIFAR-10 dataset object
        :param subset_file: '.txt' file containing indices of CIFAR-10 images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                idx = int(line.strip())
                new_samples.append(self.dataset.samples[idx])
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, target = self.samples[index]
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target



def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_file='imagenet_full_size-061417.tar.gz',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path
