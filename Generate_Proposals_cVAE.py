#!/usr/bin/env python3

"""
Module to sample a trained CVAE and propose species plantings.
Generates a GeoTIFF with top-K species IDs per pixel and a CSV summary.
"""
import os
import json
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from Train_Model_cVAE import CVAE, GEO_RASTERS, CLASS_MAP, PATCH_SIZE, LATENT_DIM, NUM_SPECIES, DEVICE

RUN_DIR    = os.path.dirname(CLASS_MAP)
MODEL_FP   = os.path.join(RUN_DIR, "cvae_fire_resilience.pt")
OUT_RASTER = os.path.join(RUN_DIR, "planting_proposals.tif")
OUT_CSV    = os.path.join(RUN_DIR, "proposal_summary.csv")
TOP_K      = 3                                                  # number of top species to propose
N_SAMPLES  = 8

"""
Load geo-rasters and initialize the CVAE model for inference.
"""
geo_ds = {k: rasterio.open(v) for k, v in GEO_RASTERS.items()}
rows, cols = geo_ds['naip'].height, geo_ds['naip'].width
num_geo = sum(ds.count for ds in geo_ds.values())
model = CVAE(num_geo, NUM_SPECIES, LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_FP, map_location=DEVICE))
model.eval()

"""
Prepare the output GeoTIFF with TOP_K bands for proposals.
"""
profile = geo_ds['naip'].profile.copy()
profile.update(
    count=TOP_K,
    dtype=rasterio.uint8,
    nodata=255,
    compress='LZW'
)
out = rasterio.open(OUT_RASTER, 'w', **profile)

"""
Sampling loop: slide patches over the grid, run Monte Carlo inference,
accumulate average logits, select top-K species per patch, and write bands.
"""
counts = np.zeros(NUM_SPECIES, dtype=np.int64)
for row in tqdm(range(0, rows, PATCH_SIZE), desc="Sampling rows"):
    for col in range(0, cols, PATCH_SIZE):
        geo_chs = []
        for ds in geo_ds.values():
            arr = ds.read(window=Window(col, row, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[None, ...]
            b, h, w = arr.shape
            pad_h = PATCH_SIZE - h
            pad_w = PATCH_SIZE - w
            if pad_h > 0 or pad_w > 0:
                arr = np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode='constant')
            geo_chs.append(arr)
        geo_np = np.concatenate(geo_chs, axis=0)[None]
        geo = torch.from_numpy(geo_np).to(DEVICE)

        logits_accum = torch.zeros(1, NUM_SPECIES, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
        with torch.no_grad():
            for _ in range(N_SAMPLES):
                dummy_lbl = torch.zeros(1, PATCH_SIZE, PATCH_SIZE, dtype=torch.long, device=DEVICE)
                c = torch.nn.functional.one_hot(dummy_lbl, NUM_SPECIES).permute(0,3,1,2).float()
                recon, _, _ = model(geo, c)
                logits_accum += recon

        probs = torch.softmax(logits_accum / N_SAMPLES, dim=1)
        topk = probs.topk(TOP_K, dim=1).indices.squeeze(0).cpu().numpy()
        counts[np.unique(topk)] += 1

        h_win = min(PATCH_SIZE, rows - row)
        w_win = min(PATCH_SIZE, cols - col)
        for k in range(TOP_K):
            band = topk[k:k+1, :h_win, :w_win]
            out.write(band, k+1, window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
out.close()
print(f"Proposals raster written to {OUT_RASTER}")

"""
Write a CSV summarizing the count of proposed pixels per species.
"""
species_json = os.path.join(RUN_DIR, "species_encoder.json")
with open(species_json) as f:
    species_map = json.load(f)
with open(OUT_CSV, 'w') as f:
    f.write("species,pixels\n")
    for sid, cnt in enumerate(counts):
        f.write(f"{species_map[str(sid)]},{int(cnt)}\n")
print(f"Summary CSV written to {OUT_CSV}")
