#!/usr/bin/env python3

"""
Module to analyze fire-resilience of species from probability rasters.
Loads species probability maps and burn severity, computes resilience scores,
and outputs CSV ranking, GeoTIFF maps, and a legend image.
"""
import os
import glob
import csv
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─────────────────────────── Placeholder paths ────────────────────────────
RUN_DIR   = "path/to/runs/eaton_aoi_convnext9"
PROB_DIR  = os.path.join(RUN_DIR, "maps")
BURN_PATH = "path/to/burn_severity.tif"
OUT_DIR   = os.path.join(RUN_DIR, "burn_analysis")
THRESHOLD = 0.9
os.makedirs(OUT_DIR, exist_ok=True)

"""
Gather all species probability rasters and load reference grid dimensions.
"""
prob_files = sorted(glob.glob(os.path.join(PROB_DIR, "*_prob.tif")))
if not prob_files:
    raise FileNotFoundError(f"No probability rasters found in {PROB_DIR}")
with rasterio.open(prob_files[0]) as ref_ds:
    H, W = ref_ds.height, ref_ds.width
    ref_profile = ref_ds.profile.copy()
    ref_transform = ref_ds.transform

"""
Load burn severity raster and sample values aligned to species grid.
Vectorized sampling to build mask of lightly burned areas.
"""
with rasterio.open(BURN_PATH) as burn_ds:
    burn_raw = burn_ds.read(1)
    burn_transform = burn_ds.transform
js, is_ = np.meshgrid(np.arange(W), np.arange(H))
is_flat, js_flat = is_.ravel(), js.ravel()
xs_flat, ys_flat = xy(ref_transform, is_flat, js_flat, offset='center')
rows_b_flat, cols_b_flat = rowcol(burn_transform, xs_flat, ys_flat)
rows_b = np.clip(rows_b_flat, 0, burn_raw.shape[0] - 1).astype(int).reshape(H, W)
cols_b = np.clip(cols_b_flat, 0, burn_raw.shape[1] - 1).astype(int).reshape(H, W)
mask = burn_raw[rows_b, cols_b] > THRESHOLD

"""
Compute raw mean probability for each species within the mask.
Results stored in species_scores as (name, raw_mean).
"""
species_scores = []
for fp in prob_files:
    sp = os.path.basename(fp).replace('_prob.tif', '')
    with rasterio.open(fp) as ds:
        prob = ds.read(1)
    if prob.shape != (H, W):
        raise ValueError(f"Shape mismatch for {sp}: {prob.shape} vs {(H,W)}")
    mean_p = float(np.nanmean(prob[mask])) if mask.any() else 0.0
    species_scores.append((sp, mean_p))

"""
Normalize raw resilience scores to [0,1].
If all scores equal, produces zeros.
"""
scores = np.array([s for _, s in species_scores], dtype=np.float32)
min_s, max_s = scores.min(), scores.max()
norm_scores = (scores - min_s) / (max_s - min_s) if max_s > min_s else np.zeros_like(scores)

"""
Build normalized float map and classification RGB map.
Select highest probability species per pixel within mask.
"""
grid_norm = np.zeros((H, W), dtype=np.float32)
max_prob = np.zeros((H, W), dtype=np.float32)
classification_idx = np.full((H, W), -1, dtype=int)
for idx, _ in enumerate(tqdm(species_scores, desc="Classifying pixels")):
    with rasterio.open(prob_files[idx]) as ds:
        prob = ds.read(1)
    new_peak = mask & (prob > max_prob)
    max_prob[new_peak] = prob[new_peak]
    grid_norm[new_peak] = norm_scores[idx]
    classification_idx[new_peak] = idx

"""
Write CSV ranking of species by normalized resilience score.
Includes raw mean and normalized score columns.
"""
summary_csv = os.path.join(OUT_DIR, "species_resilience_ranking.csv")
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["species", "mean_probability", "normalized_score"]);
    for (sp, raw), norm in zip(species_scores, norm_scores):
        writer.writerow([sp, raw, norm])
print(f"Wrote normalized resilience ranking to {summary_csv}")

"""
Write normalized float GeoTIFF for most probable species map.
Pixels unclassified set to 0.
"""
out_profile = ref_profile.copy()
out_profile.update(count=1, dtype=rasterio.float32, compress='LZW', nodata=0.0)
out_fp = os.path.join(OUT_DIR, "most_likely_species_normalized.tif")
with rasterio.open(out_fp, 'w', **out_profile) as dst:
    dst.write(grid_norm, 1)
print(f"Wrote normalized species map to {out_fp}")

"""
Generate RGB classification map using a categorical colormap.
Writes a 3-band uint8 GeoTIFF.
"""
cmap = plt.get_cmap('tab20', len(species_scores))
palette = (cmap(np.arange(len(species_scores)))[:, :3] * 255).astype(np.uint8)
rgb = np.zeros((3, H, W), dtype=np.uint8)
for idx in range(len(species_scores)):
    mask_i = (classification_idx == idx)
    for band in range(3):
        rgb[band, mask_i] = palette[idx, band]
out_profile_rgb = ref_profile.copy()
out_profile_rgb.update(count=3, dtype=rasterio.uint8, compress='LZW', nodata=0)
out_fp_rgb = os.path.join(OUT_DIR, "species_classification_rgb.tif")
with rasterio.open(out_fp_rgb, 'w', **out_profile_rgb) as dst:
    dst.write(rgb)
print(f"Wrote RGB species classification map to {out_fp_rgb}")

"""
Generate legend image mapping species names to colors.
Saves a PNG with species labels.
"""
legend_fp = os.path.join(OUT_DIR, "species_legend.png")
patches = [Patch(color=palette[i]/255.0, label=species_scores[i][0]) for i in range(len(species_scores))]
fig, ax = plt.subplots(figsize=(4, len(species_scores)*0.3))
ax.legend(handles=patches, loc='center', ncol=1, frameon=False)
ax.axis('off')
fig.savefig(legend_fp, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Wrote species legend to {legend_fp}")
