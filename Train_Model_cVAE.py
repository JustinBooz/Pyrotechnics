#!/usr/bin/env python3

"""
Script to train a conditional variational autoencoder (CVAE) on geo-rasters and class map.
Builds a Dataset from original environmental rasters and burn severity map, defines cVAE architecture, and runs the training loop.
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ─────────────────────────── paths & settings ────────────────────────────
RUN_DIR     = "path/to/runs/eaton_aoi_convnext9"
GEO_RASTERS = {
    'naip':   "path/to/naip_2022.tif",
    'elev':   "path/to/elevation.tif",
    'slope':  "path/to/slope.tif",
    'aspect': "path/to/aspect.tif",
    'dist':   "path/to/WD.tif",
    'burn':   "path/to/burn_severity.tif",
}
CLASS_MAP   = os.path.join(RUN_DIR, "burn_analysis/most_likely_species_normalized.tif")
PATCH_SIZE  = 64
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 50
LATENT_DIM  = 128
NUM_SPECIES = 31
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Dataset that extracts non-overlapping patches from geo-rasters and class map.
"""
class CVAEDataset(Dataset):
    def __init__(self, geo_rasters, class_map, patch_size):
        self.geo_ds    = {k: rasterio.open(v) for k, v in geo_rasters.items()}
        self.class_ds  = rasterio.open(class_map)
        self.patch     = patch_size
        self.nrows     = self.geo_ds['naip'].height
        self.ncols     = self.geo_ds['naip'].width
        self.positions = []
        stride = patch_size
        for i in range(0, self.nrows - stride + 1, stride):
            for j in range(0, self.ncols - stride + 1, stride):
                self.positions.append((i, j))

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i0, j0 = self.positions[idx]
        i = min(i0, max(self.nrows - self.patch, 0))
        j = min(j0, max(self.ncols - self.patch, 0))
        geo_chs = []
        for ds in self.geo_ds.values():
            arr = ds.read(window=Window(j, i, self.patch, self.patch)).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[None, ...]
            _, h, w = arr.shape
            pad_h = self.patch - h
            pad_w = self.patch - w
            if pad_h > 0 or pad_w > 0:
                arr = np.pad(arr,
                             ((0, 0), (0, pad_h), (0, pad_w)),
                             mode='constant')
            geo_chs.append(arr)
        geo = np.concatenate(geo_chs, axis=0)
        labels = self.class_ds.read(1, window=Window(j, i, self.patch, self.patch)).astype(np.int64)
        h, w = labels.shape
        if h != self.patch or w != self.patch:
            pad_h = self.patch - h
            pad_w = self.patch - w
            labels = np.pad(labels,
                            ((0, pad_h), (0, pad_w)),
                            mode='constant', constant_values=-1)
        geo = (geo - geo.mean()) / (geo.std() + 1e-6)
        return torch.from_numpy(geo), torch.from_numpy(labels)

"""
cVAE with convolutional encoder and decoder.
Encodes geoinput and label one hot into latent space and reconstructs labels.
"""
class CVAE(nn.Module):
    def __init__(self, num_geo, num_classes, latent_dim):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(num_geo + num_classes, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        hidden = 128 * (PATCH_SIZE // 4) * (PATCH_SIZE // 4)
        self.fc_mu       = nn.Linear(hidden, latent_dim)
        self.fc_logvar   = nn.Linear(hidden, latent_dim)
        self.dec_fc      = nn.Linear(latent_dim + num_geo * PATCH_SIZE * PATCH_SIZE, hidden)
        self.dec_conv    = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 4, 2, 1)
        )

    def encode(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = self.enc_conv(h).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z, x):
        geo_flat = x.flatten(1)
        h        = torch.cat([z, geo_flat], dim=1)
        h        = self.dec_fc(h).view(-1, 128, PATCH_SIZE//4, PATCH_SIZE//4)
        return self.dec_conv(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z, x), mu, logvar

"""
Compute CVAE loss: cross-entropy reconstruction plus weighted KLD.
Ignores padded pixels with label=-1..
"""
def loss_fn(recon_logits, labels, mu, logvar):
    recon_loss = nn.CrossEntropyLoss(ignore_index=-1)(recon_logits, labels)
    kld        = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 1e-3 * kld, recon_loss, kld

"""
Train the CVAE: build dataset, loader, and optimizer, then run epochs.
Saves the final model state_dict to RUN_DIR.
"""
def train():
    ds      = CVAEDataset(GEO_RASTERS, CLASS_MAP, PATCH_SIZE)
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    num_geo = sum(raster.count for raster in ds.geo_ds.values())
    model   = CVAE(num_geo, NUM_SPECIES, LATENT_DIM).to(DEVICE)
    scaler  = GradScaler()
    opt     = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        total_loss, total_recon, total_kld = 0, 0, 0
        model.train()
        for geo, labels in tqdm(loader, desc=f"Epoch {epoch}"):
            geo, labels = geo.to(DEVICE), labels.to(DEVICE)
            c          = nn.functional.one_hot(labels, NUM_SPECIES).permute(0,3,1,2).float()
            opt.zero_grad()
            with autocast():
                recon, mu, logvar = model(geo, c)
                loss, recon_l, kld = loss_fn(recon, labels, mu, logvar)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kld   += kld.item()
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}, recon={total_recon/len(loader):.4f}, kld={total_kld/len(loader):.4f}")
    ckpt = os.path.join(RUN_DIR, "cvae_fire_resilience.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Model saved to {ckpt}\n")

if __name__ == '__main__':
    train()
