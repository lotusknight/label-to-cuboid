# Label to Cuboid Service

Dockerized FastAPI inference service that converts image(s) + prompt into 3D cuboid JSON using:

- SAM 3 (prompted segmentation)
- Depth Pro (metric depth + focal length)
- Geometric lifting (mask + depth -> OBB cuboid)

```
images + "chair" -> SAM 3 -> Depth Pro -> geometric lifting -> cuboid JSON
```

## What This Service Exposes

- `POST /infer` accepts one or more images in a single request
- `GET /health` reports service readiness and configured batch size
- Structured JSON logs suitable for Docker log collectors

## API

### `POST /infer`

Multipart form fields:

- `images`: list of image files (`jpg`, `jpeg`, `png`)
- `prompt`: text query (for example `chair`)
- `confidence_threshold`: optional float in `[0, 1]`

Example:

```bash
curl -X POST "http://localhost:8000/infer" \
  -F "prompt=chair" \
  -F "confidence_threshold=0.3" \
  -F "images=@/data/photo1.jpg" \
  -F "images=@/data/photo2.jpg"
```

Response:

```json
{
  "prompt": "chair",
  "total_cuboids": 4,
  "results": [
    {
      "image_index": 0,
      "filename": "photo1.jpg",
      "count": 3,
      "cuboids": [
        {
          "object_index": 0,
          "label": "chair",
          "confidence": 0.87,
          "center": [1.23, 0.45, 3.67],
          "dimensions": [0.55, 0.82, 0.50],
          "rotation_matrix": [[0.98, -0.17, 0.01], [0.17, 0.98, 0.03], [-0.02, -0.03, 1.00]],
          "rotation_quaternion": [0.996, 0.001, -0.015, 0.085],
          "corners_3d": [[0.95, 0.04, 3.42], "..."]
        }
      ],
      "error": null
    }
  ]
}
```

If one image fails, that image result contains an `error` string and the rest of the batch still returns.

### `GET /health`

```json
{
  "status": "ok",
  "models_loaded": true,
  "gpu_batch_size": 2
}
```

## Batch Behavior

The endpoint accepts any number of images up to `MAX_IMAGES_PER_REQUEST`. Internally:

1. Inputs are chunked by `GPU_BATCH_SIZE`
2. Each chunk runs model inference
3. Results are merged and returned as a list

Tune `GPU_BATCH_SIZE` by GPU memory:

- `1` for low VRAM and safest behavior
- `2` as a common default for 16 GB GPUs
- `4+` for larger GPUs

## Configuration

Environment variables:

- `LOG_LEVEL` (default: `INFO`)
- `CONFIDENCE_THRESHOLD` (default: `0.3`)
- `GPU_BATCH_SIZE` (default: `1`)
- `MAX_IMAGES_PER_REQUEST` (default: `32`)
- `HF_ENDPOINT` (recommended in CN: `https://hf-mirror.com`)
- `HF_TOKEN` (required for gated models such as `facebook/sam3`)
- `SAM3_CHECKPOINT_PATH` (local `.pt` file path, skips HF download when set)
- `DEVICE` (default: `cuda`)
- `MODEL_DTYPE` (default: `float16`)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)

## Download SAM3 Weights

SAM3 is a gated model. You can download from a community mirror instead:

```bash
mkdir -p models
wget -O models/sam3.pt "https://hf-mirror.com/1038lab/sam3/resolve/main/sam3.pt"
```

Then set the env var to use the local file:

```bash
export SAM3_CHECKPOINT_PATH=./models/sam3.pt
```

## Run With Docker Compose

```bash
docker compose up --build
```

Service runs on `http://localhost:8000`. Docker mounts `./models` as read-only at `/models`.

## Run Locally (Without Docker)

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
unzip -q sam3-main.zip -d /tmp
pip install /tmp/sam3-main
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If you do not have `sam3-main.zip` locally, you can still install it directly:

```bash
pip install "sam3 @ https://codeload.github.com/facebookresearch/sam3/zip/refs/heads/main"
```

You can verify mirror connectivity with:

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "from transformers import DepthProImageProcessorFast; DepthProImageProcessorFast.from_pretrained('apple/DepthPro-hf')"
```

For SAM3 (gated model), request access on HuggingFace and provide a token:

```bash
export HF_TOKEN=hf_xxx
HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=$HF_TOKEN python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('facebook/sam3')"
```

## Implementation Details

- Geometric lifting utilities are in `app/geometric_lifting.py`
- Endpoints and lifecycle are in `app/main.py`
- Model orchestration and batch flow are in `app/pipeline.py`
- Full research/algorithm document: `PLAN_SAM3_DEPTH_CUBOID.md`

## Smoke Test Script

Use `scripts/smoke_test.sh` to send multi-image requests, validate response shape, and
generate cuboid overlay images.

Example (`gaosu-2`, prompts `car` and `truck`):

```bash
PROMPTS=car,truck BASE_URL=http://localhost:8000 \
  ./scripts/smoke_test.sh ./gaosu-2
```

Outputs:

- `smoke_outputs/response_car.json`
- `smoke_outputs/response_truck.json`
- `smoke_outputs/*_car_cuboid.png`
- `smoke_outputs/*_truck_cuboid.png`

Optional visualization params:

- `VIS_FX` and `VIS_FY` override projection focal length for overlay rendering
- `CONFIDENCE_THRESHOLD` controls request threshold
- `OUTPUT_DIR` sets output directory

## License

MIT
