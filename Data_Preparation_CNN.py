from __future__ import annotations
# ─────────────────────────── Paths & parameters ────────────────────────────
CSV_PATH   = "path/to/iNaturalist_Complete_Project.csv"  # update this to your CSV file
NAIP_PATH  = "path/to/naip_2022.tif"
ELEV_PATH  = "path/to/elevation.tif"
SLOPE_PATH = "path/to/slope.tif"
ASPECT_PATH= "path/to/aspect.tif"
DISTW_PATH = "path/to/WD.tif"

PATCH_SIZE = 128          # pixels (@ 0.6m)
BLOCK_DEG  = 0.002        # grid cell size for spatial split
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.20
TEST_FRAC  = 0.0

NORM_JSON  = "runs/ConvNeXt‑Tiny/norm_stats.json"    # output path for normalization stats

# ----------------------------------------------------------------------------

import os, random, numpy as np, torch
import pandas as pd, geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import box
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import json

# reproduce randomness
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def _sample_patch(ds: rasterio.DatasetReader, lon: float, lat: float, patch: int) -> np.ndarray | None:
    """Try to grab a square patch around lon/lat. Returns None near edges, if can't sample."""
    try:
        r, c = rowcol(ds.transform, lon, lat)
    except Exception:
        return None
    h = patch // 2
    rs, re = r - h, r + h
    cs, ce = c - h, c + h
    if rs < 0 or cs < 0 or re > ds.height or ce > ds.width:
        return None
    return ds.read(window=((rs, re), (cs, ce))).astype(np.float32)

def _aspect_sin_cos(aspect_patch: np.ndarray):
    """Convert aspect degrees to sin and cos compnents for better modeling."""
    rad = np.deg2rad(aspect_patch)
    return np.sin(rad), np.cos(rad)

def _make_grid(bounds, cell):
    """Generate a lat/lon grid covering bounds, with cells of given size. Handy for spatial split."""
    minx, miny, maxx, maxy = bounds
    polys, x = [], minx
    while x < maxx:
        y = miny
        while y < maxy:
            polys.append(box(x, y, x+cell, y+cell))
            y += cell
        x += cell
    return gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")

def _spatial_split(gdf, train, val, test, cell):
    """Assign each point in gdf to train/val/test based on random grid cell assignment."""
    grid = _make_grid(gdf.total_bounds, cell)
    grid["cid"] = range(len(grid))
    joined = gpd.sjoin(gdf, grid, how="left", predicate="within")
    cells = np.random.permutation(joined["cid"].unique())
    n = len(cells)
    tr = cells[:int(n*train)]
    va = cells[int(n*train):int(n*(train+val))]
    te = cells[int(n*(train+val)):]
    lut = {c: "train" for c in tr} | {c: "val" for c in va} | {c: "test" for c in te}
    return joined["cid"].map(lut).values

class GeoPatchDataset(Dataset):
    """Torch Dataset for patch arrays and labels. Converts numpy arrays to tensors for CNN trainning."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(
    csv_path: str,
    naip_path: str,
    cont_rasters: dict[str,str],
    patch_size: int = 128,
    block_deg: float = 0.01,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15
    ):
    """Main functon: load csv, sample patches, standarise continuous bands, and split datasets."""
    # read CSV → GeoDataFrame
    df  = pd.read_csv(csv_path)
    gdf = gpd.GeoDataFrame(df,
           geometry=gpd.points_from_xy(df.longitude, df.latitude),
           crs="EPSG:4326")

    # open rasters
    naip_ds   = rasterio.open(naip_path)
    elev_ds   = rasterio.open(cont_rasters["elev"])
    slope_ds  = rasterio.open(cont_rasters["slope"])
    aspect_ds = rasterio.open(cont_rasters["aspect"])
    dist_ds   = rasterio.open(cont_rasters["distw"])

    # sample patches & record surviving indices
    patches, labels, keep_idx = [], [], []
    for idx, row in gdf.iterrows():
        naip = _sample_patch(naip_ds,   row.longitude, row.latitude, patch_size)
        if naip is None: continue
        elev   = _sample_patch(elev_ds,   row.longitude, row.latitude, patch_size)
        slope  = _sample_patch(slope_ds,  row.longitude, row.latitude, patch_size)
        aspect = _sample_patch(aspect_ds, row.longitude, row.latitude, patch_size)
        dist   = _sample_patch(dist_ds,   row.longitude, row.latitude, patch_size)
        if any(p is None for p in (elev, slope, aspect, dist)):  continue

        # normalise NAIP and augment
        naip = naip / 255.0
        if random.random() < 0.3:
            naip[:3] += np.random.normal(0, 0.05, size=naip[:3].shape)
            naip = np.clip(naip, 0.0, 1.0).astype(np.float32)
        sinA, cosA = _aspect_sin_cos(aspect)  # spatial orientatn encoded
        full = np.concatenate([naip, elev, slope, sinA, cosA, dist], axis=0)
        if random.random() < 0.5:
            full = full[:, :, ::-1]  # horizontal flip augmntation
        patches.append(full)
        labels.append(row.classname)
        keep_idx.append(idx)

    # alignment check to ensure labels match patches
    gdf_valid = gdf.iloc[keep_idx].reset_index(drop=True)
    assert all(
        gdf_valid.classname.iloc[i] == labels[i]
        for i in range(len(labels))
    ), "Patch / label mis-alignment detected!"

    # stack & encode labels
    X = np.stack(patches, dtype=np.float32)   # → [N,9,H,W]
    y = np.array(labels)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    # spatial split on filtered points
    splits = _spatial_split(
        gdf_valid, train_frac, val_frac, test_frac, block_deg
    )
    tr = np.where(splits=="train")[0]
    va = np.where(splits=="val"  )[0]
    te = np.where(splits=="test")[0]

    # standardise continuous bands using training stats
    cont = slice(4,9)
    mean = X[tr][:,cont].mean(axis=(0,2,3))
    std  = X[tr][:,cont].std (axis=(0,2,3)) + 1e-6
    X[:,cont] = (X[:,cont] - mean[None,:,None,None]) / std[None,:,None,None]

    norm_stats = {
        "mean": mean.tolist(),
        "std":  std.tolist(),
        "channels": ["elev","slope","aspect_sin","aspect_cos","distw"]
    }

    # build and return datasets
    train_ds = GeoPatchDataset(X[tr],    y_enc[tr])
    val_ds   = GeoPatchDataset(X[va],    y_enc[va])
    test_ds  = GeoPatchDataset(X[te],    y_enc[te]) if len(te) else None

    return train_ds, val_ds, test_ds, encoder, norm_stats, (tr,va,te)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(NORM_JSON), exist_ok=True)
    train_ds, val_ds, test_ds, enc, stats, _ = prepare_data(
        CSV_PATH, NAIP_PATH,
        {"elev":  ELEV_PATH,
         "slope": SLOPE_PATH,
         "aspect":ASPECT_PATH,
         "distw": DISTW_PATH},
        PATCH_SIZE, BLOCK_DEG, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    with open(NORM_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print("✓ datasets built:")
    if test_ds is None:
        print(f"  train: {len(train_ds):>6d}  |  val: {len(val_ds):>5d}")
    else:
        print(f"  train: {len(train_ds):>6d}  |  val: {len(val_ds):>5d}  |"
              f"  test: {len(test_ds):>5d}")
    print(f"  norm stats → {NORM_JSON}")
