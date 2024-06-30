import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
# from vit_module import vit_tiny, MaskCollator  # Adjust the import based on your file structure
from src.models.vision_transformer import vit_tiny, vit_predictor
from src.masks.multiblock import MaskCollator as MBMaskCollator

# CIFAR-10 dataset and data loaders
def get_cifar10_dataloaders(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=MBMaskCollator())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=MBMaskCollator())

    return train_loader, val_loader

# Joint-Embedding Predictive Architecture (I-JEPA)
class IJEPA(nn.Module):
    def __init__(self, encoder, predictor, embed_dim):
        super(IJEPA, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.embed_dim = embed_dim

    def forward(self, x, context_mask, target_mask):
        context_embeds = self.encoder(x, masks=context_mask)
        with torch.no_grad():
            target_embeds = self.encoder(x, masks=target_mask)
        predictions = self.predictor(context_embeds, context_mask, target_mask)
        return predictions, target_embeds

# Simple training loop for I-JEPA
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for imgs, masks_enc, masks_pred in dataloader:
        imgs = imgs[0].to(device)
        masks_enc = [mask.to(device) for mask in masks_enc]
        masks_pred = [mask.to(device) for mask in masks_pred]
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, target_embeds = model(imgs, masks_enc, masks_pred)
        loss = F.mse_loss(predictions, target_embeds)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    batch_size = 64
    num_workers = 4
    lr = 0.001
    num_epochs = 10
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    train_loader, val_loader = get_cifar10_dataloaders(batch_size, num_workers)

    encoder = vit_tiny(patch_size=16, img_size=[224])
    predictor = vit_predictor(num_patches=16, embed_dim=192)
    model = IJEPA(encoder, predictor, embed_dim=192).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')

if __name__ == "__main__":
    main()

