# Label to Cuboid

Convert a 2D image + text label into 3D oriented bounding boxes (cuboids) using SAM 3 + Depth Pro + geometric lifting.

```
Image + "chair"  →  SAM 3 (masks)  →  Depth Pro (metric depth)  →  3D Cuboids (JSON)
```

## How It Works

1. **SAM 3** segments all instances of the text label in the image (e.g. all "chairs")
2. **Depth Pro** estimates metric depth (meters) and camera focal length from the raw image — no camera metadata needed
3. **Geometric lifting** back-projects masked depth pixels to 3D and fits an oriented bounding box (OBB) per object

## Requirements

- Linux 64-bit with NVIDIA GPU (16 GB+ VRAM recommended)
- Python 3.12+
- CUDA 12.6+

## Quick Start

```bash
# 1. Create environment
conda create -n cuboid-pipeline python=3.12 -y
conda activate cuboid-pipeline

# 2. Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install SAM 3
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e . && cd ..
huggingface-cli login  # SAM 3 weights require access approval

# 4. Install Depth Pro (HuggingFace version, NVIDIA-optimized)
pip install "transformers>=4.48" accelerate

# 5. Install extras
pip install scipy

# 6. Run
python label_to_cuboid.py --image photo.jpg --label "chair"
```

## Output

```json
{
    "label": "chair",
    "confidence": 0.87,
    "center": [1.23, 0.45, 3.67],
    "dimensions": [0.55, 0.82, 0.50],
    "rotation_quaternion": [0.996, 0.001, -0.015, 0.085],
    "corners_3d": [[0.95, 0.04, 3.42], "...8 corners"]
}
```

Coordinate system: camera frame (X-right, Y-down, Z-forward), units in meters.

## Performance

| Stage | Time | Notes |
|-------|------|-------|
| SAM 3 | ~1-2 sec | Finds all instances |
| Depth Pro | ~0.3 sec | With float16 on NVIDIA |
| Geometric lifting | ~10 ms | Per object, CPU only |
| **Total** | **~1.5-2.5 sec** | Per image |

## Documentation

See [PLAN_SAM3_DEPTH_CUBOID.md](PLAN_SAM3_DEPTH_CUBOID.md) for the complete implementation plan including:
- Full pipeline code (copy-paste ready)
- Geometric lifting math (back-projection, PCA-based OBB)
- Linux + NVIDIA GPU setup guide (step by step)
- Docker deployment
- Low-VRAM mode (8-12 GB GPUs)
- Troubleshooting
- UniDepth V2 as a fallback depth model

## License

MIT
