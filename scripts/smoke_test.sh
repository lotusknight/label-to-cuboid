#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
PROMPTS="${PROMPTS:-chair}"
OUTPUT_DIR="${OUTPUT_DIR:-./smoke_outputs}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.3}"
VIS_FX="${VIS_FX:-0}"
VIS_FY="${VIS_FY:-0}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image_or_dir1> [image_or_dir2 ...]"
  echo "Example: PROMPTS=car,truck $0 ./gaosu-2"
  echo "Example: PROMPTS=car,truck $0 ./a.jpg ./b.jpg"
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

IFS=',' read -r -a prompt_list <<< "$PROMPTS"
for prompt in "${prompt_list[@]}"; do
  prompt="$(echo "$prompt" | xargs)"
  if [[ -z "$prompt" ]]; then
    continue
  fi

  tmp_response="$(mktemp)"
  tmp_status="$(mktemp)"
  cleanup() {
    rm -f "$tmp_response" "$tmp_status"
  }
  trap cleanup EXIT

  curl_args=(
    -sS
    -o "$tmp_response"
    -w "%{http_code}"
    -X POST
    "${BASE_URL}/infer"
    -F "prompt=${prompt}"
    -F "confidence_threshold=${CONFIDENCE_THRESHOLD}"
  )

  for image in "${images[@]}"; do
    curl_args+=(-F "images=@${image}")
  done

  http_code="$(curl "${curl_args[@]}")"
  echo "$http_code" > "$tmp_status"

  response_file="${OUTPUT_DIR}/response_${prompt}.json"
  cp "$tmp_response" "$response_file"

  python3 - "$tmp_response" "$tmp_status" "$prompt" "$OUTPUT_DIR" "$VIS_FX" "$VIS_FY" "${images[@]}" <<'PY'
import json
import pathlib
import sys
from typing import Iterable

from PIL import Image, ImageDraw

response_path = pathlib.Path(sys.argv[1])
status_path = pathlib.Path(sys.argv[2])
expected_prompt = sys.argv[3]
output_dir = pathlib.Path(sys.argv[4])
vis_fx = float(sys.argv[5])
vis_fy = float(sys.argv[6])
image_paths = [pathlib.Path(p) for p in sys.argv[7:]]

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

if payload["prompt"] != expected_prompt:
    raise SystemExit(
        f"[FAIL] prompt mismatch: got={payload['prompt']!r} expected={expected_prompt!r}"
    )

if not isinstance(payload["total_cuboids"], int):
    raise SystemExit("[FAIL] total_cuboids must be int")

results = payload["results"]
if not isinstance(results, list):
    raise SystemExit("[FAIL] results must be a list")
if len(results) != len(image_paths):
    raise SystemExit(
        f"[FAIL] results length mismatch: got={len(results)} expected={len(image_paths)}"
    )

required_item = {"image_index", "filename", "count", "cuboids", "error"}
for idx, item in enumerate(results):
    if not isinstance(item, dict):
        raise SystemExit(f"[FAIL] results[{idx}] must be object")
    missing_item = required_item - set(item)
    if missing_item:
        raise SystemExit(f"[FAIL] results[{idx}] missing keys: {sorted(missing_item)}")
    if not isinstance(item["image_index"], int):
        raise SystemExit(f"[FAIL] results[{idx}].image_index must be int")
    if not isinstance(item["filename"], str):
        raise SystemExit(f"[FAIL] results[{idx}].filename must be str")
    if not isinstance(item["count"], int):
        raise SystemExit(f"[FAIL] results[{idx}].count must be int")
    if not isinstance(item["cuboids"], list):
        raise SystemExit(f"[FAIL] results[{idx}].cuboids must be list")
    if item["error"] is not None and not isinstance(item["error"], str):
        raise SystemExit(f"[FAIL] results[{idx}].error must be null or str")


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


COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 165, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 255, 0),
]

# Corner layout (signs array order):
#   0=(-,-,-) 1=(-,-,+) 2=(-,+,-) 3=(-,+,+)
#   4=(+,-,-) 5=(+,-,+) 6=(+,+,-) 7=(+,+,+)
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

        for ci, cuboid in enumerate(item["cuboids"]):
            color = COLORS[ci % len(COLORS)]
            label = cuboid.get("label", expected_prompt)
            conf = cuboid.get("confidence", 0.0)

            bbox = cuboid.get("bbox_2d")
            if isinstance(bbox, list) and len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                draw_over.rectangle([x0, y0, x1, y1], fill=color + (40,))
                draw_main.rectangle([x0, y0, x1, y1], outline=color, width=2)
                draw_main.text((x0 + 2, y0 - 14), f"{label}:{conf:.2f}", fill=(255, 255, 255))

            corners = cuboid.get("corners_3d")
            if isinstance(corners, list) and len(corners) == 8:
                pts = [project_corner(c, width, height, focal=item_fx) for c in corners]
                if not any(p is None for p in pts):
                    for a, b in front_edges:
                        draw_main.line((pts[a], pts[b]), fill=color, width=3)
                    for a, b in back_edges:
                        draw_main.line((pts[a], pts[b]), fill=color, width=2)
                    for a, b in depth_edges:
                        draw_main.line((pts[a], pts[b]), fill=color, width=2)

        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        out_name = f"{source.stem}_{expected_prompt}_cuboid.png"
        out_path = output_dir / out_name
        image.save(out_path)
        generated += 1

print("[PASS] Smoke test passed.")
print(
    f"[INFO] prompt={payload['prompt']!r} images={len(image_paths)} "
    f"total_cuboids={payload['total_cuboids']} rendered={generated}"
)
print(f"[INFO] response={response_path}")
print(f"[INFO] output_dir={output_dir}")
PY
  rm -f "$tmp_response" "$tmp_status"
  trap - EXIT
done

echo "[DONE] all prompts finished: $PROMPTS"
