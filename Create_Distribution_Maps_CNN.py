#!/usr/bin/env python3

"""
Module to infer species distribution maps using ConvNeXt9Ch model.
Slides a window over input rasters and produces GeoTIFF maps for each species.
Uses a boundary-aware Dataset and standardizes channels for model input.
"""
import os
import json
import math
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from Model_Definition_CNN import ConvNeXt9Ch
from Data_Preparation_CNN import _aspect_sin_cos

# ──────────── Placeholder paths & settings ─────────────
RUN_DIR      = "path/to/runs/eaton_aoi_convnext9"
CHECKPOINT   = os.path.join(RUN_DIR, "best_model.pt")
SPECIES_JSON = os.path.join(RUN_DIR, "species_encoder.json")
NORM_JSON    = os.path.join(RUN_DIR, "norm_stats.json")
NAIP_PATH    = "path/to/naip_2022.tif"
ELEV_PATH    = "path/to/elevation.tif"
SLOPE_PATH   = "path/to/slope.tif"
ASPECT_PATH  = "path/to/aspect.tif"
DISTW_PATH   = "path/to/WD.tif"

OUT_DIR      = os.path.join(RUN_DIR, "maps")
STRIDE       = 64
PATCH_SIZE   = 128
BATCH_SIZE   = 64
NUM_WORKERS  = 0  # Windows-friendly; avoid pickling issues

# Open rasters once
rasters = {
    'naip':   rasterio.open(NAIP_PATH),  # 4-band NAIP (R,G,B,NIR)
    'elev':   rasterio.open(ELEV_PATH),
    'slope':  rasterio.open(SLOPE_PATH),
    'aspect': rasterio.open(ASPECT_PATH),
    'dist':   rasterio.open(DISTW_PATH)
}
naip_ds = rasters['naip']
H, W = naip_ds.height, naip_ds.width
rows = math.ceil((H - PATCH_SIZE) / STRIDE) + 1
cols = math.ceil((W - PATCH_SIZE) / STRIDE) + 1

def read_and_pad(name, bands, normalize_divisor=None):
    """
    Read band data from raster for current window and pad to PATCH_SIZE.
    'name' and 'bands' recieves the raster key and band index(s).
    If 'normalize_divisor' is provided, divides data by it.
    Retuns array shape (C, PATCH_SIZE, PATCH_SIZE).
    """
    window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
    arr = rasters[name].read(bands, window=window).astype(np.float32)
    if normalize_divisor is not None:
        arr /= normalize_divisor
    if arr.ndim == 2:
        arr = arr[None, ...]
    c, h, w = arr.shape
    pad_h = PATCH_SIZE - h
    pad_w = PATCH_SIZE - w
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
    return arr

class WindowDataset(Dataset):
    """
    Dataset that slides a window over all rasters and yields patches.
    Each __getitem__ retuns (patch, x, y) for a window.
    Clamps coordintes and pads tiles at edges to avoid out-of-bounds.
    """
    def __len__(self):
        return rows * cols

    def __getitem__(self, idx):
        global x, y
        r = idx // cols
        c = idx % cols
        y0 = r * STRIDE
        x0 = c * STRIDE
        y = min(y0, max(H - PATCH_SIZE, 0))
        x = min(x0, max(W - PATCH_SIZE, 0))

        naip   = read_and_pad('naip',   [1,2,3,4], normalize_divisor=255.0)
        elev   = read_and_pad('elev',   1)
        slope  = read_and_pad('slope',  1)
        aspect = read_and_pad('aspect', 1)
        dist   = read_and_pad('dist',   1)

        sinA, cosA = _aspect_sin_cos(aspect)
        patch = np.concatenate([naip, elev, slope, sinA, cosA, dist], axis=0)
        return patch, x, y

def collate_fn(batch_list):
    """
    collate function to batch patches and standaradize continuous channels.
    Loads normalization stats and applies (arr - mean)/std on channels [4:].
    """
    patches, xs, ys = zip(*batch_list)
    arr = np.stack(patches, axis=0)
    with open(NORM_JSON, 'r') as f:
        norm = json.load(f)
    mean = np.array(norm['mean'], dtype=np.float32)
    std  = np.array(norm['std'], dtype=np.float32)
    arr[:, 4:] = (arr[:, 4:] - mean[None, :, None, None]) / std[None, :, None, None]
    return torch.from_numpy(arr), torch.tensor(xs), torch.tensor(ys)

def main():
    """
    Main inferenc pipelin: loads model, species list, and DataLoader;
    runs inference and writes GeoTIFFs to OUT_DIR.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(SPECIES_JSON, 'r') as f:
        species = json.load(f)

    model = ConvNeXt9Ch(len(species)).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = WindowDataset()
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS,
                         collate_fn=collate_fn)

    prob_maps = np.zeros((len(species), H, W), dtype=np.float32)
    counts    = np.zeros((H, W), dtype=np.uint16)
    softmax   = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch, xs, ys in tqdm(loader, total=len(loader)):
            batch = batch.to(device)
            out   = softmax(model(batch)).cpu().numpy()
            for i in range(out.shape[0]):
                xi, yi = int(xs[i]), int(ys[i])
                prob_maps[:, yi:yi+PATCH_SIZE, xi:xi+PATCH_SIZE] += out[i][:, None, None]
                counts   [yi:yi+PATCH_SIZE, xi:xi+PATCH_SIZE] += 1

    counts    = np.clip(counts, 1, None)
    prob_maps /= counts[None]

    profile = naip_ds.profile.copy()
    profile.update(count=1, dtype=rasterio.float32, compress='LZW')
    for idx, sp in enumerate(species):
        out_fp = os.path.join(OUT_DIR, f"{sp}_prob.tif")
        with rasterio.open(out_fp, 'w', **profile) as dst:
            dst.write(prob_maps[idx], 1)

    print(f"Wrote {len(species)} probability maps to {OUT_DIR}")

if __name__ == '__main__':
    main()
