"""GPU-accelerated cloud / cloud-shadow masking via OmniCloudMask (DPIRD-DMA).

Replaces the Sentinel-2 SCL-based mask as the source of pixel validity: SCL
is still recorded on samples for reference/QA, but ``valid_px`` is decided
by the OmniCloudMask (OCM) class for the pixel. See METHODS.md for the
rationale and the parameter choices made below.
"""
from pathlib import Path

import numpy as np
import torch
from omnicloudmask import predict_from_array

# OmniCloudMask output classes.
OCM_CLEAR = 0
OCM_THICK_CLOUD = 1
OCM_THIN_CLOUD = 2
OCM_SHADOW = 3

# Strict clear-sky: only OCM_CLEAR counts as a valid pixel.
OCM_INVALID_CLASSES = (OCM_THICK_CLOUD, OCM_THIN_CLOUD, OCM_SHADOW)

# Inference settings (OmniCloudMask published defaults).
OCM_PATCH_SIZE = 1000
OCM_PATCH_OVERLAP = 300
OCM_BATCH_SIZE = 1
OCM_MODEL_VERSION = None  # None = latest

# Model weights are downloaded on first use; cache them inside the repo
# (gitignored) rather than an out-of-project system cache directory.
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache" / "omnicloudmask"


def inference_device_and_dtype():
    """Auto-detect the best available torch device.

    fp16 is used on CUDA GPUs for speed; CPU and MPS stay at fp32 (fp16 is
    slow/unsupported on most CPUs and MPS has partial fp16 support).
    """
    if torch.cuda.is_available():
        return "cuda", "fp16"
    if torch.backends.mps.is_available():
        return "mps", "fp32"
    return "cpu", "fp32"


def compute_ocm_class(red, green, nir, no_data_value=0.0):
    """Run OmniCloudMask on one Red/Green/NIR image.

    ``red``/``green``/``nir`` are 2D numpy arrays of identical shape, in the
    same units as loaded from the STAC assets (raw DN). Returns a (H, W)
    uint8 array of OmniCloudMask classes (0=clear, 1=thick cloud,
    2=thin cloud, 3=shadow).
    """
    stacked = np.stack([red, green, nir], axis=0).astype(np.float32)
    device, dtype = inference_device_and_dtype()

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pred = predict_from_array(
        stacked,
        patch_size=OCM_PATCH_SIZE,
        patch_overlap=OCM_PATCH_OVERLAP,
        batch_size=OCM_BATCH_SIZE,
        inference_device=device,
        inference_dtype=dtype,
        no_data_value=no_data_value,
        model_version=OCM_MODEL_VERSION,
        destination_model_dir=MODEL_CACHE_DIR,
    )
    return pred[0].astype("uint8")


def ocm_clear_mask(red, green, nir, no_data_value=0.0):
    """Boolean clear-sky mask (True = valid pixel) plus the raw class array."""
    ocm_class = compute_ocm_class(red, green, nir, no_data_value=no_data_value)
    clear = ~np.isin(ocm_class, OCM_INVALID_CLASSES)
    return clear, ocm_class


def warm_model_cache():
    """Force the OmniCloudMask model weights to download, synchronously.

    Must be called once in the main process before spinning up parallel
    scene-processing workers. omnicloudmask downloads its weights lazily
    on first use; if multiple worker processes hit an empty cache at the
    same time, they race to write the same file and most of them crash
    with a spurious "No such file or directory" on the partially-written
    weights (observed at max_workers=32 on a cold cache - not a fluke,
    reproduced consistently). Calling this first, single-threaded, means
    the file already exists by the time workers start, so none of them
    ever trigger a download.
    """
    dummy = np.zeros((64, 64), dtype="float32")
    compute_ocm_class(dummy, dummy, dummy)
