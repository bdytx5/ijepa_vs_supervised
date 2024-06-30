# Import Required Modules
import math
import numpy as np
import torch
import torch.nn as nn
import fastcore.all as fc
from PIL import Image
from functools import partial
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

# Set parameters
img_size = 192
patch_size = 32

# Load and visualize an image using COCO val data
dataset = CocoDetection(root='../coco/val2017', annFile='../coco/annotations/instances_val2017.json')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
imgs, _ = next(iter(data_loader))

# Define the standard transforms
def transforms():
    return Compose([RandomResizedCrop(size=img_size, scale=[0.4, 1], ratio=[0.75, 1.33], interpolation=2),
                    RandomHorizontalFlip(p=0.5),
                    ToTensor()])

# Load and transform the image
def load_img(img, transforms):
    return transforms(img)

transform = transforms()
img = load_img(imgs[0], transform)
img = img.unsqueeze(0)
img.shape # torch.Size([1, 3, 192, 192])

# Set up input data for auto-regression
imgp = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size)
imgp.shape # torch.Size([36, 3, 32, 32])

# Randomly mask 50% of the patches
tokens = imgp.shape[0]
mask_ratio = 0.5
mask_count = int(tokens * mask_ratio)
mask_idx = torch.randperm(tokens)[:mask_count]
mask = torch.zeros(tokens).long()
mask[mask_idx] = 1

# Mask token and PatchEmbed
from timm.models.vision_transformer import PatchEmbed

pe = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=768, norm_layer=nn.LayerNorm)
embed = pe(img)
embed.shape # torch.Size([1, 36, 768])

# Replace masked patches with mask token
mask_token = nn.Parameter(torch.zeros(1, 1, 768))
w = mask.unsqueeze(-1).type_as(mask_token)
embed_masked = (embed * (1. - w) + mask_token * w)
embed_masked.shape # torch.Size([1, 36, 768])

# ViT Encoder
class ViTEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        super(ViTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Instantiate ViT encoder
vit_encoder = ViTEncoder(embed_dim=768, depth=12, num_heads=12)
encoded = vit_encoder(embed_masked)
encoded.shape # torch.Size([1, 36, 768])

# Decoder to reconstruct image patches
class SimpleDecoder(nn.Module):
    def __init__(self, patch_size, embed_dim, in_chans):
        super(SimpleDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=patch_size**2 * in_chans, kernel_size=1),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L**0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return self.decoder(x)

decoder = SimpleDecoder(patch_size=patch_size, embed_dim=768, in_chans=3)
out = decoder(encoded)
out.shape # torch.Size([1, 3, 192, 192])

# Reshape the output to match the original patch shape
out_patches = out.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size)
out_patches.shape # torch.Size([36, 3, 32, 32])

# Calculate the loss using L1 loss
x = imgp[mask.bool(), ...]
x_rec = out_patches[mask.bool(), ...]
loss = torch.nn.functional.l1_loss(x, x_rec, reduction='none').mean()
print(loss)
