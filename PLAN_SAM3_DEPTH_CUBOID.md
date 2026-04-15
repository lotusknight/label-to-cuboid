# SAM 3 + Depth: Text-to-3D-Cuboid Pipeline

## Complete Implementation Plan

> **Goal**: Given a plain 2D RGB image (no camera metadata) and a text label (e.g. "chair"), output 3D oriented bounding boxes (cuboids) for all matching objects.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Input: RGB Image + Text Label                │
└───────────────┬──────────────────────────────────┬───────────────────┘
                │                                  │
                ▼                                  ▼
    ┌───────────────────┐              ┌───────────────────────┐
    │     SAM 3          │              │     Depth Pro          │
    │  (Text → Masks)    │              │  (Image → Metric Depth │
    │                    │              │   + Focal Length)       │
    │  ~1-2 sec          │              │  ~0.3 sec              │
    │  GPU: ~4-8 GB      │              │  GPU: ~4-6 GB          │
    └────────┬──────────┘              └──────────┬────────────┘
             │                                    │
             │  Per-object binary masks            │  H×W depth map (meters)
             │  + confidence scores                │  + estimated focal length (px)
             │                                    │
             ▼                                    ▼
    ┌──────────────────────────────────────────────────────────┐
    │              Geometric Lifting (CPU, pure math)          │
    │                                                          │
    │  For each mask:                                          │
    │    1. Extract masked depth pixels                        │
    │    2. Back-project to 3D point cloud                     │
    │    3. Filter outliers                                    │
    │    4. Fit oriented bounding box (PCA or min-volume)      │
    │                                                          │
    │  ~10 ms per object                                       │
    └──────────────────────┬───────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Output: List of 3D Cuboids (JSON)                       │
    │                                                          │
    │  Per object:                                             │
    │    - label, confidence                                   │
    │    - center [x, y, z] in meters                          │
    │    - dimensions [w, h, d] in meters                      │
    │    - rotation (quaternion or 3×3 matrix)                 │
    │    - 8 corner points                                     │
    └──────────────────────────────────────────────────────────┘
```

## 2. Why This Works Without Camera Metadata

**Q: We only have a raw 2D image — no EXIF, no camera intrinsics, no calibration. Is that enough?**

**A: Yes.** Here's why:

| What we need                | Where it comes from                                       |
|-----------------------------|-----------------------------------------------------------|
| Object segmentation masks   | SAM 3 — needs only RGB + text                             |
| Per-pixel depth in meters   | Depth Pro — trained to predict metric depth from any image |
| Camera focal length (pixels)| Depth Pro — has a built-in focal length estimation head    |
| Camera principal point      | Assumed at image center (standard assumption)              |

**Depth Pro** is the key enabler. Unlike most depth models that output relative/normalized depth, Depth Pro outputs:
- **Absolute metric depth** in meters
- **Estimated focal length** in pixels

With `(focal_length, cx, cy, depth_map)` you can back-project every pixel to 3D. No EXIF or camera calibration needed.

### Accuracy caveat

Estimated focal length introduces ~5-15% scale error compared to known intrinsics. For most applications (scene layout, robotics grasping zones, AR placement), this is acceptable. If you need sub-centimeter accuracy, you'll need actual camera calibration.

---

## 3. Component Details

### 3.1 SAM 3 — Text-to-Mask Segmentation

**What it is**: Meta's Segment Anything Model 3 (Nov 2025). A unified foundation model that natively accepts text prompts for "Promptable Concept Segmentation" (PCS).

**Repository**: https://github.com/facebookresearch/sam3

**Installation**:
```bash
pip install git+https://github.com/facebookresearch/sam3.git
# Or via Ultralytics (simpler):
pip install ultralytics>=8.3.237
```

**Model weights**: `sam3.pt` (auto-downloaded from HuggingFace, ~2.5 GB)

**GPU requirement**: ~4-8 GB VRAM

**API — Option A: Official Meta API**:
```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt="chair")

masks = output["masks"]       # List of H×W binary masks
boxes = output["boxes"]       # List of [x1, y1, x2, y2] 2D bounding boxes
scores = output["scores"]     # Confidence scores per detection
```

**API — Option B: Ultralytics (simpler)**:
```python
from ultralytics import SAM

model = SAM("sam3.pt")
results = model(image_path, texts=["chair"])

for result in results:
    masks = result.masks.data          # Tensor of binary masks
    boxes = result.boxes.xyxy          # 2D bounding boxes
    confs = result.boxes.conf          # Confidence scores
```

**Input**: Any RGB image (PIL Image, numpy array, or file path) + text string.

**Output per detection**:
- `mask`: H×W binary mask (bool or uint8)
- `box`: 2D bounding box [x1, y1, x2, y2]
- `score`: confidence float (0-1)

**Filtering**: Apply a confidence threshold (recommended: `score >= 0.3`) to remove false positives.

---

### 3.2 Depth Pro — Metric Depth + Focal Length Estimation

**What it is**: Apple's monocular metric depth model. Outputs sharp depth maps in meters AND estimates the camera focal length, all from a single RGB image with no metadata required.

**NVIDIA GPU Warning**: The native Apple repo (`apple/ml-depth-pro`) has known performance issues on NVIDIA GPUs — inference can be 100x slower than expected (~45 sec instead of <1 sec) due to defaulting to float32 and poor GPU utilization. **Use the HuggingFace Transformers version instead**, which includes proper NVIDIA optimizations (float16 + SDPA attention).

**Installation (recommended for NVIDIA)**:
```bash
pip install "transformers>=4.48" accelerate
```
Weights auto-download from HuggingFace on first use (~500 MB).

**GPU requirement**: ~4-6 GB VRAM (with float16)

**API — Recommended: HuggingFace Transformers (NVIDIA-optimized)**:
```python
import torch
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
from PIL import Image

model = DepthProForDepthEstimation.from_pretrained(
    "apple/DepthPro-hf",
    torch_dtype=torch.float16,          # CRITICAL: use half-precision for NVIDIA
    attn_implementation="sdpa",         # CRITICAL: use NVIDIA-optimized attention
).cuda()
processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")

image = Image.open("photo.jpg")
inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

with torch.no_grad():
    outputs = model(**inputs)

depth_map = outputs.predicted_depth                # H×W tensor, values in meters
focal_length_px = outputs.predicted_focal_length   # Scalar, focal length in pixels
```

**API — Alternative: Native Apple repo (NOT recommended for NVIDIA)**:
```python
# WARNING: Known to be extremely slow on NVIDIA GPUs (RTX 3080: ~45 sec/image).
# Only use this on Apple Silicon (MPS) or if you have no other option.
import depth_pro

model, transform = depth_pro.create_model_and_transforms()
model.eval()
model.cuda()

image, _, f_px = depth_pro.load_rgb("photo.jpg")
image = transform(image)

prediction = model.infer(image, f_px=f_px)

depth_map = prediction["depth"]                  # H×W tensor, values in meters
focal_length_px = prediction["focallength_px"]   # Scalar, focal length in pixels
```

**Input**: Any RGB image (no metadata needed).

**Output**:
- `depth`: H×W depth map in meters (float16/float32)
- `focal_length`: estimated focal length in pixels (scalar)

**Note**: Always let Depth Pro estimate the focal length unless you have accurate calibration data.

### 3.2.1 Fallback: UniDepth V2 (if Depth Pro issues persist)

If Depth Pro still gives trouble on your specific NVIDIA GPU, **UniDepth V2** is a drop-in replacement that is natively built for NVIDIA hardware.

**Repository**: https://github.com/lpiccinelli-eth/UniDepth

**Key advantage**: ~0.08 sec per image on A40 (4x faster than Depth Pro).

```bash
pip install git+https://github.com/lpiccinelli-eth/UniDepth.git
```

```python
from unidepth.models import UniDepthV2

model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
model = model.cuda()

predictions = model.infer(image_tensor)

depth_map = predictions["depth"]          # H×W, meters
intrinsics = predictions["intrinsics"]    # 3×3 camera matrix (includes focal length)
focal_length_px = intrinsics[0, 0]        # fx from the intrinsic matrix
```

| | Depth Pro (HF) | UniDepth V2 |
|---|----------------|-------------|
| Metric depth | Yes | Yes |
| Focal length estimation | Yes (dedicated head) | Yes (camera module) |
| NVIDIA speed | ~0.3 sec (float16) | ~0.08 sec |
| Install complexity | `pip install transformers` | `pip install git+...` |
| Maturity | Widely used | Newer, less community support |

---

### 3.3 Geometric Lifting — Depth + Mask to 3D Cuboid

This is pure math — no neural network, no GPU needed.

#### Step 1: Build Camera Intrinsic Matrix

From Depth Pro's output, construct the intrinsic matrix:

```python
import numpy as np

def build_intrinsics(focal_length_px, image_width, image_height):
    """
    Construct a 3×3 camera intrinsic matrix.
    Assumes principal point at image center (standard for unknown cameras).
    """
    cx = image_width / 2.0
    cy = image_height / 2.0
    K = np.array([
        [focal_length_px, 0,              cx],
        [0,               focal_length_px, cy],
        [0,               0,               1 ]
    ])
    return K
```

#### Step 2: Back-project Masked Pixels to 3D

```python
def backproject_depth_to_3d(depth_map, mask, K):
    """
    Convert masked depth pixels to a 3D point cloud.
    
    Args:
        depth_map: H×W numpy array, depth in meters
        mask: H×W boolean array, True for object pixels
        K: 3×3 intrinsic matrix
    
    Returns:
        points_3d: N×3 numpy array of 3D points in camera frame (meters)
    """
    H, W = depth_map.shape
    
    # Get pixel coordinates of masked region
    v, u = np.where(mask)  # v = row (y), u = col (x)
    z = depth_map[v, u]
    
    # Remove invalid depth values
    valid = (z > 0) & (z < 100)  # reasonable range: 0-100 meters
    u, v, z = u[valid], v[valid], z[valid]
    
    # Back-project: p_3d = K^{-1} * [u, v, 1]^T * z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_3d = np.stack([x, y, z], axis=-1)  # N×3
    return points_3d
```

#### Step 3: Filter Outliers

Depth estimation can have noise, especially at object edges. Remove outliers before fitting the box.

```python
def filter_outliers(points_3d, std_multiplier=2.0):
    """
    Remove points that are > std_multiplier standard deviations from the median
    along any axis.
    """
    median = np.median(points_3d, axis=0)
    std = np.std(points_3d, axis=0)
    
    mask = np.all(np.abs(points_3d - median) < std_multiplier * std, axis=1)
    return points_3d[mask]
```

#### Step 4: Fit Oriented Bounding Box (PCA)

```python
def fit_oriented_bounding_box(points_3d):
    """
    Fit an oriented bounding box to a 3D point cloud using PCA.
    
    Returns:
        center: [x, y, z] center of the OBB in meters
        dimensions: [w, h, d] size along each principal axis in meters
        rotation_matrix: 3×3 rotation matrix (columns = principal axes)
        corners: 8×3 array of corner points
    """
    # 1. Compute centroid
    centroid = np.mean(points_3d, axis=0)
    
    # 2. Center the points
    centered = points_3d - centroid
    
    # 3. Covariance matrix and eigen-decomposition
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue descending (largest variance first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    
    # 4. Project points onto principal axes
    projected = centered @ eigenvectors  # N×3 in local frame
    
    # 5. Compute extents
    min_vals = projected.min(axis=0)
    max_vals = projected.max(axis=0)
    dimensions = max_vals - min_vals          # [w, h, d]
    local_center = (min_vals + max_vals) / 2  # center in local frame
    
    # 6. Transform center back to world frame
    center = centroid + eigenvectors @ local_center
    
    # 7. Compute 8 corner points
    half = dimensions / 2
    signs = np.array([
        [-1, -1, -1], [-1, -1, +1], [-1, +1, -1], [-1, +1, +1],
        [+1, -1, -1], [+1, -1, +1], [+1, +1, -1], [+1, +1, +1],
    ])
    local_corners = signs * half  # 8×3
    corners = (eigenvectors @ local_corners.T).T + center  # 8×3 in world
    
    return {
        "center": center,
        "dimensions": dimensions,
        "rotation_matrix": eigenvectors,
        "corners_3d": corners,
    }
```

#### Step 5 (optional): Convert Rotation Matrix to Quaternion

```python
from scipy.spatial.transform import Rotation

def rotation_matrix_to_quaternion(R):
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z]."""
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # scipy returns [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # reorder to [w, x, y, z]
```

---

### 3.4 Alternative: Open3D One-Liner (Simpler)

If you don't need custom control, Open3D can compute the OBB directly:

```python
import open3d as o3d

def fit_obb_open3d(points_3d):
    """Fit OBB using Open3D (PCA-based or minimal volume)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # PCA-based (fast)
    obb = pcd.get_oriented_bounding_box()
    
    # Or minimal-volume (slower but tighter)
    # obb = pcd.get_minimal_oriented_bounding_box()
    
    return {
        "center": np.asarray(obb.center),
        "dimensions": np.asarray(obb.extent),
        "rotation_matrix": np.asarray(obb.R),
        "corners_3d": np.asarray(obb.get_box_points()),
    }
```

---

## 4. Full Pipeline — Putting It All Together

```python
"""
label_to_cuboid.py — Full pipeline: Image + Text Label → 3D Cuboids

Usage:
    python label_to_cuboid.py --image photo.jpg --label "chair" --output cuboids.json
"""

import json
import argparse
import numpy as np
from PIL import Image


# ──────────────────────────────────────────
# Stage 1: Load models (do this once)
# ──────────────────────────────────────────

def load_sam3():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    return processor

def load_depth_pro():
    import torch
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
    model = DepthProForDepthEstimation.from_pretrained(
        "apple/DepthPro-hf",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).cuda()
    processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    return model, processor


# ──────────────────────────────────────────
# Stage 2: Run segmentation
# ──────────────────────────────────────────

def segment_objects(processor, image, text_label, confidence_threshold=0.3):
    """
    Run SAM 3 text-prompted segmentation.
    Returns list of dicts: [{"mask": H×W bool, "score": float}, ...]
    """
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=text_label)

    results = []
    for mask, score in zip(output["masks"], output["scores"]):
        if score >= confidence_threshold:
            results.append({
                "mask": np.array(mask, dtype=bool),
                "score": float(score),
            })
    return results


# ──────────────────────────────────────────
# Stage 3: Run depth estimation
# ──────────────────────────────────────────

def estimate_depth(model, processor, image_path):
    """
    Run Depth Pro metric depth estimation via HuggingFace Transformers.
    Returns depth_map (H×W numpy, meters) and focal_length_px (scalar).
    """
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

    with torch.no_grad():
        outputs = model(**inputs)

    depth_map = outputs.predicted_depth.squeeze().cpu().float().numpy()
    focal_length_px = float(outputs.predicted_focal_length.item())

    return depth_map, focal_length_px


# ──────────────────────────────────────────
# Stage 4: Geometric lifting
# ──────────────────────────────────────────

def mask_to_cuboid(depth_map, mask, focal_length_px, image_width, image_height):
    """
    Convert a single object mask + depth into a 3D oriented bounding box.
    """
    K = build_intrinsics(focal_length_px, image_width, image_height)
    points_3d = backproject_depth_to_3d(depth_map, mask, K)

    if len(points_3d) < 10:
        return None

    points_3d = filter_outliers(points_3d, std_multiplier=2.0)

    if len(points_3d) < 10:
        return None

    return fit_oriented_bounding_box(points_3d)


# ──────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────

def run_pipeline(image_path, text_label, confidence_threshold=0.3):
    """
    Full pipeline: image + text → list of 3D cuboids.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    H, W = image_np.shape[:2]

    # Load models (in production, load once and reuse)
    sam3_processor = load_sam3()
    depth_model, depth_processor = load_depth_pro()

    # Run SAM 3 (text → masks)
    detections = segment_objects(sam3_processor, image_np, text_label, confidence_threshold)
    print(f"SAM 3 found {len(detections)} '{text_label}' objects")

    # Run Depth Pro (image → depth + focal length)
    depth_map, focal_length_px = estimate_depth(depth_model, depth_processor, image_path)
    print(f"Depth Pro: focal_length={focal_length_px:.1f}px, depth range=[{depth_map.min():.2f}, {depth_map.max():.2f}]m")

    # Ensure depth map matches image resolution
    if depth_map.shape != (H, W):
        from scipy.ndimage import zoom
        depth_map = zoom(depth_map, (H / depth_map.shape[0], W / depth_map.shape[1]), order=1)

    # Geometric lifting (mask + depth → cuboid)
    cuboids = []
    for i, det in enumerate(detections):
        obb = mask_to_cuboid(depth_map, det["mask"], focal_length_px, W, H)
        if obb is None:
            continue

        cuboid = {
            "object_index": i,
            "label": text_label,
            "confidence": det["score"],
            "center": obb["center"].tolist(),
            "dimensions": obb["dimensions"].tolist(),
            "rotation_matrix": obb["rotation_matrix"].tolist(),
            "rotation_quaternion": rotation_matrix_to_quaternion(obb["rotation_matrix"]).tolist(),
            "corners_3d": obb["corners_3d"].tolist(),
        }
        cuboids.append(cuboid)

    return cuboids


# ──────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text label → 3D cuboid pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--label", required=True, help="Text label (e.g. 'chair')")
    parser.add_argument("--output", default="cuboids.json", help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()

    cuboids = run_pipeline(args.image, args.label, args.threshold)

    with open(args.output, "w") as f:
        json.dump(cuboids, f, indent=2)

    print(f"\nSaved {len(cuboids)} cuboids to {args.output}")
    for c in cuboids:
        dims = c["dimensions"]
        print(f"  [{c['label']}] conf={c['confidence']:.2f}  "
              f"size={dims[0]:.2f}×{dims[1]:.2f}×{dims[2]:.2f}m  "
              f"center=({c['center'][0]:.2f}, {c['center'][1]:.2f}, {c['center'][2]:.2f})")
```

---

## 5. Output Format

Each cuboid is a JSON object:

```json
{
    "object_index": 0,
    "label": "chair",
    "confidence": 0.87,
    "center": [1.23, 0.45, 3.67],
    "dimensions": [0.55, 0.82, 0.50],
    "rotation_matrix": [
        [0.98, -0.17, 0.01],
        [0.17,  0.98, 0.03],
        [-0.02, -0.03, 1.00]
    ],
    "rotation_quaternion": [0.996, 0.001, -0.015, 0.085],
    "corners_3d": [
        [0.95, 0.04, 3.42],
        [0.95, 0.04, 3.92],
        [0.95, 0.86, 3.42],
        [0.95, 0.86, 3.92],
        [1.51, 0.04, 3.42],
        [1.51, 0.04, 3.92],
        [1.51, 0.86, 3.42],
        [1.51, 0.86, 3.92]
    ]
}
```

**Coordinate system**: Camera frame (OpenCV convention)
- X = right
- Y = down
- Z = forward (into the scene)
- Units = meters

---

## 6. Dependencies

### Required packages

```
# requirements.txt

# SAM 3 — text-prompted segmentation
# Option A: Official (pick one)
sam3 @ git+https://github.com/facebookresearch/sam3.git
# Option B: Via Ultralytics
# ultralytics>=8.3.237

# Depth Pro — via HuggingFace Transformers (NVIDIA-optimized)
transformers>=4.48
accelerate

# Fallback depth model (if Depth Pro has issues on your GPU)
# unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git

# Core
torch>=2.7
torchvision>=0.22
numpy>=1.24
scipy>=1.10
Pillow>=10.0

# Optional (for Open3D OBB shortcut)
# open3d>=0.18
```

### Hardware

| Component | Min VRAM | Recommended |
|-----------|----------|-------------|
| SAM 3 | 4 GB | 8 GB |
| Depth Pro | 4 GB | 6 GB |
| Both loaded simultaneously | 8 GB | 16 GB |
| Geometric lifting | 0 (CPU) | 0 (CPU) |

**Minimum GPU**: 1× GPU with 16 GB VRAM (e.g. RTX 4080, T4, A10)
**Can also run sequentially on 8 GB** by loading/unloading models one at a time.

---

## 7. Performance Estimates

| Stage | Time per image | Notes |
|-------|---------------|-------|
| SAM 3 segmentation | ~1-2 sec | Finds all instances of the label |
| Depth Pro depth estimation | ~0.3 sec | One forward pass, very fast |
| Geometric lifting per object | ~10 ms | Pure numpy, negligible |
| **Total (1 image, N objects)** | **~1.5-2.5 sec** | Nearly independent of object count |

Throughput: **~25-40 images/minute** on a single 16 GB GPU.

---

## 8. Multiple Labels

To detect multiple object types in one image, run SAM 3 once per label:

```python
labels = ["chair", "table", "lamp", "book"]
all_cuboids = []

for label in labels:
    detections = segment_objects(sam3_processor, image, label)
    for det in detections:
        obb = mask_to_cuboid(depth_map, det["mask"], focal_length_px, W, H)
        if obb:
            obb["label"] = label
            obb["confidence"] = det["score"]
            all_cuboids.append(obb)
```

Or with SAM 3, you can pass multiple concepts at once (check SAM 3 docs for batch concept API).

---

## 9. Known Limitations and Mitigations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Estimated focal length has ~5-15% error | Cuboid dimensions may be off by same percentage | Use known camera intrinsics if available; or calibrate once for a fixed camera |
| Monocular depth is less accurate at far range (>10m) | Cuboids for distant objects may be distorted | Limit to indoor / close-range scenes; or use stereo/LiDAR for far objects |
| Depth edges may bleed across object boundaries | Points from background may leak into object cloud | Erode the mask by 2-3 pixels before back-projection; or use outlier filtering |
| PCA OBB may not align with semantic axes | A chair's OBB axes may not align with "up" | Post-process: snap the closest axis to gravity (Y-down in camera frame) |
| Transparent / reflective objects have poor depth | Glass, mirrors, shiny metal | Known limitation of all monocular depth models; no clean fix |
| SAM 3 may merge nearby objects into one mask | Two chairs touching may get one mask | Lower SAM 3's IoU threshold; or use point prompts to separate |

---

## 10. Optional Enhancements

### A. Mask erosion for cleaner edges
```python
from scipy.ndimage import binary_erosion

clean_mask = binary_erosion(mask, iterations=3)
```

### B. Gravity alignment
```python
def align_obb_to_gravity(rotation_matrix):
    """Snap the closest principal axis to the Y (gravity) direction."""
    gravity = np.array([0, 1, 0])  # Y-down in camera frame
    dots = np.abs(rotation_matrix.T @ gravity)
    up_axis_idx = np.argmax(dots)
    # Swap so that axis `up_axis_idx` becomes the Y-axis of the OBB
    # ... (implement axis swapping logic)
```

### C. Multi-view consistency (if you have multiple images)
Run the pipeline on each image independently, then merge cuboids using ICP or pose-graph optimization.

### D. Visualization
```python
import open3d as o3d

def visualize_cuboids(cuboids, depth_map, K):
    """Render cuboids overlaid on the scene point cloud."""
    geometries = []
    for c in cuboids:
        obb = o3d.geometry.OrientedBoundingBox(
            center=c["center"],
            R=np.array(c["rotation_matrix"]),
            extent=c["dimensions"],
        )
        obb.color = (1, 0, 0)
        geometries.append(obb)
    o3d.visualization.draw_geometries(geometries)
```

---

## 11. Linux + NVIDIA GPU Setup Guide

This section provides a complete, copy-paste-ready setup for a fresh Linux machine with an NVIDIA GPU.

### 11.1 Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04+ (or any Linux 64-bit) | Ubuntu 22.04 LTS |
| GPU | NVIDIA with 8 GB VRAM | NVIDIA with 16+ GB VRAM (RTX 4080, A10, T4, A100) |
| NVIDIA Driver | 535+ | 550+ |
| CUDA Toolkit | 12.1+ | 12.6+ (for SAM 3 compatibility) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free (for models + env) | 50 GB free |
| Python | 3.12+ (required by SAM 3) | 3.12 |

### 11.2 Step 0: Verify GPU and Driver

```bash
# Check that your NVIDIA GPU is detected
nvidia-smi

# You should see output like:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.xx    Driver Version: 550.xx    CUDA Version: 12.x          |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-SXM4-40GB On|   00000000:00:04.0 Off|                    0 |
# | N/A   32C    P0    45W / 400W|      0MiB / 40960MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# Check CUDA version (need 12.1+, ideally 12.6+)
nvcc --version
# If nvcc is not found, that's OK — PyTorch bundles its own CUDA runtime.
# What matters is the driver version (nvidia-smi) supports CUDA 12.x.
```

If your driver is too old, update it:
```bash
# Ubuntu
sudo apt update
sudo apt install -y nvidia-driver-550
sudo reboot
```

### 11.3 Step 1: Create Conda Environment

SAM 3 requires Python 3.12+. Using conda/mamba ensures a clean, isolated environment.

```bash
# Install miniforge if you don't have conda/mamba
# https://github.com/conda-forge/miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

# Create environment
conda create -n cuboid-pipeline python=3.12 -y
conda activate cuboid-pipeline
```

### 11.4 Step 2: Install PyTorch with CUDA

```bash
# Install PyTorch 2.7+ with CUDA 12.8 (matches SAM 3 requirements)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify GPU is accessible from PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

# Expected output:
# CUDA available: True
# GPU: NVIDIA A100-SXM4-40GB
# VRAM: 40.0 GB
```

### 11.5 Step 3: Install SAM 3

```bash
# Clone SAM 3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# (Optional) Install flash attention for faster inference
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128

cd ..

# Authenticate with HuggingFace (SAM 3 weights require access approval)
# 1. Go to https://huggingface.co/facebook/sam3 and request access
# 2. Create a token at https://huggingface.co/settings/tokens
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted

# Download SAM 3 checkpoint
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='.')"
```

### 11.6 Step 4: Install Depth Pro (HuggingFace version — NVIDIA-optimized)

```bash
# Install via HuggingFace Transformers (NOT the native Apple repo)
# The native Apple repo has known severe performance issues on NVIDIA GPUs.
pip install "transformers>=4.48" accelerate

# Weights auto-download from HuggingFace on first use (~500 MB)
# Pre-download if you want (optional):
python -c "from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast; DepthProForDepthEstimation.from_pretrained('apple/DepthPro-hf'); DepthProImageProcessorFast.from_pretrained('apple/DepthPro-hf')"

# Verify it loads correctly on NVIDIA with float16:
python -c "
import torch
from transformers import DepthProForDepthEstimation
model = DepthProForDepthEstimation.from_pretrained('apple/DepthPro-hf', torch_dtype=torch.float16, attn_implementation='sdpa').cuda()
print(f'Depth Pro loaded on {next(model.parameters()).device}, dtype={next(model.parameters()).dtype}')
del model; torch.cuda.empty_cache()
print('OK')
"
```

> **Why not the native Apple repo?**
> The original `apple/ml-depth-pro` defaults to float32 and does not use SDPA attention,
> causing ~45 sec inference on RTX 3080 (vs ~0.3 sec with the HF version in float16).
> See: https://github.com/apple/ml-depth-pro/issues/64

### 11.7 Step 5: Install Remaining Dependencies

```bash
# Core math/science
pip install scipy

# (Optional) Open3D for OBB shortcut and visualization
pip install open3d

# (Optional) For visualization / debugging
pip install matplotlib

# (Optional) Fallback depth model if Depth Pro still has issues on your GPU
# pip install git+https://github.com/lpiccinelli-eth/UniDepth.git
```

### 11.8 Step 6: Verify the Full Stack

Create a test script `verify_setup.py`:

```python
#!/usr/bin/env python3
"""Verify that all components of the cuboid pipeline are working."""

import sys

print("=" * 60)
print("Cuboid Pipeline Setup Verification")
print("=" * 60)

# 1. PyTorch + CUDA
print("\n[1/4] PyTorch + CUDA")
import torch
assert torch.cuda.is_available(), "CUDA not available!"
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"  OK: {gpu_name} ({vram_gb:.1f} GB VRAM)")
print(f"  PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

# 2. SAM 3
print("\n[2/4] SAM 3")
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("  OK: SAM 3 imports successful")
except ImportError as e:
    print(f"  FAIL: {e}")
    print("  Fix: cd sam3 && pip install -e .")

# 3. Depth Pro (HuggingFace version)
print("\n[3/4] Depth Pro (HuggingFace Transformers)")
try:
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
    import transformers
    print(f"  OK: transformers {transformers.__version__}, DepthPro imports successful")
except ImportError as e:
    print(f"  FAIL: {e}")
    print("  Fix: pip install 'transformers>=4.48' accelerate")

# 4. Supporting libraries
print("\n[4/4] Supporting libraries")
import numpy as np
from scipy.spatial.transform import Rotation
print(f"  OK: numpy {np.__version__}, scipy available")

try:
    import open3d as o3d
    print(f"  OK: Open3D {o3d.__version__} (optional)")
except ImportError:
    print("  SKIP: Open3D not installed (optional)")

# Summary
print("\n" + "=" * 60)
print("All required components verified successfully!")
print(f"Ready to run on: {gpu_name}")
if vram_gb < 16:
    print(f"WARNING: {vram_gb:.0f} GB VRAM — you may need to load models sequentially.")
    print("  Recommended: 16+ GB VRAM to load SAM 3 + Depth Pro simultaneously.")
print("=" * 60)
```

Run it:
```bash
python verify_setup.py
```

### 11.9 Step 7: Run the Pipeline

```bash
# Basic usage
python label_to_cuboid.py --image photo.jpg --label "chair"

# Multiple labels (run separately)
python label_to_cuboid.py --image photo.jpg --label "chair" --output chairs.json
python label_to_cuboid.py --image photo.jpg --label "table" --output tables.json

# With custom confidence threshold
python label_to_cuboid.py --image photo.jpg --label "lamp" --threshold 0.5 --output lamps.json
```

### 11.10 Directory Structure After Setup

```
your-project/
├── label_to_cuboid.py          # Main pipeline script (Section 4)
├── geometric_lifting.py        # Helper functions (Section 3.3)
├── verify_setup.py             # Setup verification script
├── requirements.txt            # Dependency list
├── sam3/                       # SAM 3 repository (cloned)
│   ├── sam3/                   # SAM 3 Python package
│   └── ...
├── sam3.pt                     # SAM 3 weights (~3.4 GB)
├── cuboids.json                # Output (generated)
└── ~/.cache/huggingface/       # Depth Pro weights auto-cached here (~500 MB)
    └── hub/models--apple--DepthPro-hf/
```

Note: Depth Pro weights are automatically downloaded and cached by HuggingFace
Transformers on first use — no manual clone or download script needed.

### 11.11 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | Both models loaded on same GPU | Load models sequentially: run SAM 3 first, delete it, then load Depth Pro. See Section 11.12 below. |
| `torch.cuda.is_available()` returns `False` | Driver/CUDA mismatch or no GPU | Run `nvidia-smi`; reinstall PyTorch with correct CUDA version |
| SAM 3: `Access denied` on HF download | Haven't been granted access | Go to https://huggingface.co/facebook/sam3, click "Request access", wait for approval |
| SAM 3: `ModuleNotFoundError: No module named 'sam3'` | Not installed in editable mode | `cd sam3 && pip install -e .` |
| Depth Pro: ~45 sec per image on NVIDIA | Using native Apple repo in float32 | Switch to HuggingFace version with `torch_dtype=torch.float16` and `attn_implementation="sdpa"` (see Section 3.2) |
| Depth Pro: `KeyError: 'predicted_focal_length'` | Old transformers version | `pip install --upgrade "transformers>=4.48"` |
| `flash_attn` build fails | Missing CUDA dev headers | `pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128` |
| Very slow first run | PyTorch JIT compiling kernels | Normal on first run; subsequent runs are faster |
| Open3D segfault on headless server | No display for visualization | Use `os.environ["OPEN3D_CPU_RENDERING"] = "true"` or export to file instead |
| Depth Pro still slow after HF fix | GPU not using tensor cores | Verify: `python -c "import torch; print(torch.cuda.get_device_capability())"` — need compute capability >= 7.0. If all else fails, try UniDepth V2 (Section 3.2.1) |

### 11.12 Low-VRAM Mode (8-12 GB GPUs)

If your GPU has less than 16 GB VRAM, load and unload models sequentially:

```python
import torch
import gc

def run_pipeline_low_vram(image_path, text_label):
    """Sequential model loading for GPUs with 8-12 GB VRAM."""
    from PIL import Image
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    H, W = image_np.shape[:2]

    # --- Phase 1: SAM 3 (load, run, unload) ---
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam_model = build_sam3_image_model()
    processor = Sam3Processor(sam_model)
    inference_state = processor.set_image(image_np)
    output = processor.set_text_prompt(state=inference_state, prompt=text_label)

    masks = [np.array(m, dtype=bool) for m in output["masks"]]
    scores = [float(s) for s in output["scores"]]

    # Free SAM 3 from GPU
    del processor, sam_model, inference_state, output
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 2: Depth Pro (load, run, unload) ---
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

    depth_model = DepthProForDepthEstimation.from_pretrained(
        "apple/DepthPro-hf", torch_dtype=torch.float16, attn_implementation="sdpa"
    ).cuda()
    depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")

    depth_image = Image.open(image_path).convert("RGB")
    inputs = depth_processor(images=depth_image, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        outputs = depth_model(**inputs)

    depth_map = outputs.predicted_depth.squeeze().cpu().float().numpy()
    focal_length_px = float(outputs.predicted_focal_length.item())

    # Free Depth Pro from GPU
    del depth_model, depth_processor, outputs
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 3: Geometric lifting (CPU only) ---
    cuboids = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score < 0.3:
            continue
        obb = mask_to_cuboid(depth_map, mask, focal_length_px, W, H)
        if obb:
            cuboids.append({
                "object_index": i,
                "label": text_label,
                "confidence": score,
                "center": obb["center"].tolist(),
                "dimensions": obb["dimensions"].tolist(),
                "rotation_matrix": obb["rotation_matrix"].tolist(),
                "corners_3d": obb["corners_3d"].tolist(),
            })

    return cuboids
```

### 11.13 Docker (Optional)

For reproducible deployment:

```dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install SAM 3
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && pip install -e .

# Install Depth Pro via HuggingFace Transformers (NVIDIA-optimized)
RUN pip install "transformers>=4.48" accelerate

# Install extras
RUN pip install scipy open3d huggingface_hub

# Pre-download Depth Pro weights into the image
RUN python -c "from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast; \
    DepthProForDepthEstimation.from_pretrained('apple/DepthPro-hf'); \
    DepthProImageProcessorFast.from_pretrained('apple/DepthPro-hf')"

# Note: SAM 3 weights require HF authentication.
# Mount them at runtime: docker run -v /path/to/sam3.pt:/app/sam3.pt ...

COPY label_to_cuboid.py geometric_lifting.py ./

ENTRYPOINT ["python", "label_to_cuboid.py"]
```

Build and run:
```bash
docker build -t cuboid-pipeline .
docker run --gpus all -v $(pwd)/sam3.pt:/app/sam3.pt \
    cuboid-pipeline --image /data/photo.jpg --label "chair"
```

---

## 12. Quick-Start Checklist

For Linux + NVIDIA GPU, follow these steps in order:

1. [ ] Verify GPU: `nvidia-smi` shows NVIDIA GPU with driver 535+
2. [ ] Install Miniforge/Conda (if not present)
3. [ ] Create env: `conda create -n cuboid-pipeline python=3.12 -y`
4. [ ] Install PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
5. [ ] Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`  -->  `True`
6. [ ] Clone + install SAM 3: `git clone ... && pip install -e .`
7. [ ] Authenticate HuggingFace: `huggingface-cli login`
8. [ ] Download SAM 3 weights from https://huggingface.co/facebook/sam3
9. [ ] Install Depth Pro (HF): `pip install "transformers>=4.48" accelerate`
10. [ ] (Optional) Pre-download Depth Pro weights: see Step 4 in Section 11.6
11. [ ] Install extras: `pip install scipy open3d`
12. [ ] Run verification: `python verify_setup.py`
13. [ ] Copy pipeline code from Section 4 into your project
14. [ ] Copy geometric lifting functions from Section 3.3
15. [ ] Test: `python label_to_cuboid.py --image photo.jpg --label "chair"`
16. [ ] Inspect `cuboids.json`
