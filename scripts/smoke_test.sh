#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
PROMPTS="${PROMPTS:-chair}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-./smoke_outputs/${TIMESTAMP}}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.3}"
HEADING_MODE="${HEADING_MODE:-}"
VIS_FX="${VIS_FX:-0}"
VIS_FY="${VIS_FY:-0}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image_or_dir1> [image_or_dir2 ...]"
  echo "Example: PROMPTS=car,truck $0 ./gaosu-2"
  exit 1
fi

images=()
for input_path in "$@"; do
  if [[ -d "$input_path" ]]; then
    while IFS= read -r -d '' file; do
      images+=("$file")
    done < <(find "$input_path" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) -print0)
  elif [[ -f "$input_path" ]]; then
    images+=("$input_path")
  else
    echo "Input not found: $input_path"
    exit 1
  fi
done

if [[ ${#images[@]} -eq 0 ]]; then
  echo "No images found in inputs."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

tmp_response="$(mktemp)"
tmp_status="$(mktemp)"
cleanup() { rm -f "$tmp_response" "$tmp_status"; }
trap cleanup EXIT

curl_args=(
  -sS
  -o "$tmp_response"
  -w "%{http_code}"
  -X POST
  "${BASE_URL}/infer"
  -F "prompt=${PROMPTS}"
  -F "confidence_threshold=${CONFIDENCE_THRESHOLD}"
)

if [[ -n "$HEADING_MODE" ]]; then
  curl_args+=(-F "heading_mode=${HEADING_MODE}")
fi

for image in "${images[@]}"; do
  curl_args+=(-F "images=@${image}")
done

http_code="$(curl "${curl_args[@]}")"
echo "$http_code" > "$tmp_status"

response_file="${OUTPUT_DIR}/response.json"
cp "$tmp_response" "$response_file"

python3 - "$tmp_response" "$tmp_status" "$OUTPUT_DIR" "$VIS_FX" "$VIS_FY" "${images[@]}" <<'PY'
import json
import pathlib
import sys
from typing import Iterable

from PIL import Image, ImageDraw

response_path = pathlib.Path(sys.argv[1])
status_path = pathlib.Path(sys.argv[2])
output_dir = pathlib.Path(sys.argv[3])
vis_fx = float(sys.argv[4])
vis_fy = float(sys.argv[5])
image_paths = [pathlib.Path(p) for p in sys.argv[6:]]

status = status_path.read_text().strip()
if status != "200":
    raise SystemExit(f"[FAIL] HTTP status is {status}, expected 200")

try:
    payload = json.loads(response_path.read_text())
except json.JSONDecodeError as exc:
    raise SystemExit(f"[FAIL] Response is not valid JSON: {exc}") from exc

required_top = {"prompt", "total_cuboids", "results"}
missing_top = required_top - set(payload)
if missing_top:
    raise SystemExit(f"[FAIL] Missing top-level keys: {sorted(missing_top)}")

if not isinstance(payload["total_cuboids"], int):
    raise SystemExit("[FAIL] total_cuboids must be int")

results = payload["results"]
if not isinstance(results, list):
    raise SystemExit("[FAIL] results must be a list")
if len(results) != len(image_paths):
    raise SystemExit(
        f"[FAIL] results length mismatch: got={len(results)} expected={len(image_paths)}"
    )

labels = payload.get("labels", [payload["prompt"]])

LABEL_COLORS = {}
PALETTE = [
    (255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 165, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 255, 0),
]
for i, lbl in enumerate(labels):
    LABEL_COLORS[lbl] = PALETTE[i % len(PALETTE)]


def project_corner(corner: Iterable[float], width: int, height: int, focal: float = 0) -> tuple[float, float] | None:
    x, y, z = corner
    if z <= 1e-6:
        return None
    fx = focal if focal > 0 else (vis_fx if vis_fx > 0 else float(max(width, height)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return (u, v)


front_edges = [(0, 2), (2, 6), (6, 4), (4, 0)]
back_edges  = [(1, 3), (3, 7), (7, 5), (5, 1)]
depth_edges = [(0, 1), (2, 3), (4, 5), (6, 7)]

output_dir.mkdir(parents=True, exist_ok=True)
generated = 0

for item in results:
    image_index = item["image_index"]
    if image_index < 0 or image_index >= len(image_paths):
        continue

    source = image_paths[image_index]
    item_fx = item.get("focal_length_px", 0)
    with Image.open(source).convert("RGB") as image:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw_main = ImageDraw.Draw(image)
        draw_over = ImageDraw.Draw(overlay)
        width, height = image.size

        for cuboid in item["cuboids"]:
            label = cuboid.get("label", "?")
            conf = cuboid.get("confidence", 0.0)
            color = LABEL_COLORS.get(label, (200, 200, 200))

            corners = cuboid.get("corners_3d")
            if not isinstance(corners, list) or len(corners) != 8:
                continue
            pts = [project_corner(c, width, height, focal=item_fx) for c in corners]
            if any(p is None for p in pts):
                continue

            face = [pts[0], pts[2], pts[6], pts[4]]
            draw_over.polygon(face, fill=color + (40,))

            for a, b in front_edges:
                draw_main.line((pts[a], pts[b]), fill=color, width=3)
            for a, b in back_edges:
                draw_main.line((pts[a], pts[b]), fill=color, width=2)
            for a, b in depth_edges:
                draw_main.line((pts[a], pts[b]), fill=color, width=2)

            tx, ty = pts[4]
            draw_main.text((tx + 2, ty - 14), f"{label}:{conf:.2f}", fill=(255, 255, 255))

        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        out_name = f"{source.stem}_cuboid.png"
        out_path = output_dir / out_name
        image.save(out_path)
        generated += 1

print("[PASS] Smoke test passed.")
print(
    f"[INFO] labels={labels} images={len(image_paths)} "
    f"total_cuboids={payload['total_cuboids']} rendered={generated}"
)
print(f"[INFO] response={response_path}")
print(f"[INFO] output_dir={output_dir}")
PY

echo "[DONE] finished: $PROMPTS"
