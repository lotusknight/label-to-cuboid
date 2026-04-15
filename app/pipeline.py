from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from app.config import Settings
from app.geometric_lifting import (
    backproject_depth_to_3d,
    build_intrinsics,
    filter_outliers,
    fit_oriented_bounding_box,
    rotation_matrix_to_quaternion,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceImage:
    filename: str
    image: Image.Image


def _chunked(items: list[InferenceImage], chunk_size: int) -> Iterable[list[InferenceImage]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


class SamBackend:
    def segment(
        self, images: list[Image.Image], prompt: str, threshold: float
    ) -> list[list[dict[str, np.ndarray | float]]]:
        raise NotImplementedError


class Sam3OfficialBackend(SamBackend):
    def __init__(self) -> None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model()
        self.processor = Sam3Processor(model)

    def segment(
        self, images: list[Image.Image], prompt: str, threshold: float
    ) -> list[list[dict[str, np.ndarray | float]]]:
        results: list[list[dict[str, np.ndarray | float]]] = []
        for image in images:
            image_np = np.array(image.convert("RGB"))
            state = self.processor.set_image(image_np)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
            detections: list[dict[str, np.ndarray | float]] = []
            for mask, score in zip(output.get("masks", []), output.get("scores", [])):
                conf = float(score)
                if conf < threshold:
                    continue
                detections.append({"mask": np.array(mask, dtype=bool), "score": conf})
            results.append(detections)
        return results


class Sam3TransformersBackend(SamBackend):
    def __init__(self, model_id: str, device: str, torch_dtype: "torch.dtype") -> None:
        import torch
        from transformers import AutoModelForMaskGeneration, AutoProcessor

        self.torch = torch
        self.device = device
        self.model = AutoModelForMaskGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def segment(
        self, images: list[Image.Image], prompt: str, threshold: float
    ) -> list[list[dict[str, np.ndarray | float]]]:
        prompts = [prompt for _ in images]
        inputs = self.processor(images=images, text=prompts, return_tensors="pt").to(
            self.device
        )
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [(img.height, img.width) for img in images]
        processed = self.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, target_sizes=target_sizes
        )

        all_results: list[list[dict[str, np.ndarray | float]]] = []
        for sample in processed:
            sample_results: list[dict[str, np.ndarray | float]] = []
            segmentation = sample.get("segmentation")
            segments_info = sample.get("segments_info", [])
            if segmentation is None:
                all_results.append(sample_results)
                continue

            segmentation_np = np.array(segmentation)
            for seg in segments_info:
                seg_id = seg.get("id")
                score = float(seg.get("score", 1.0))
                if seg_id is None or score < threshold:
                    continue
                mask = segmentation_np == seg_id
                if np.any(mask):
                    sample_results.append({"mask": mask, "score": score})
            all_results.append(sample_results)

        return all_results


class DepthBackend:
    def __init__(self, model_id: str, device: str, torch_dtype: "torch.dtype") -> None:
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

        self.torch = torch
        self.processor = DepthProImageProcessorFast.from_pretrained(model_id)
        self.model = DepthProForDepthEstimation.from_pretrained(
            model_id, torch_dtype=torch_dtype, attn_implementation="sdpa"
        ).to(device)
        self.device = device
        self.torch_dtype = torch_dtype

    def estimate(self, images: list[Image.Image]) -> tuple[list[np.ndarray], list[float]]:
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        depths = outputs.predicted_depth.detach().cpu().float().numpy()
        focals_raw = outputs.predicted_focal_length.detach().cpu().float().numpy()

        depth_maps: list[np.ndarray] = []
        focal_lengths: list[float] = []
        for i, image in enumerate(images):
            depth = depths[i]
            if depth.shape != (image.height, image.width):
                h_scale = image.height / depth.shape[0]
                w_scale = image.width / depth.shape[1]
                depth = zoom(depth, (h_scale, w_scale), order=1)
            depth_maps.append(depth)
            focal_lengths.append(float(np.atleast_1d(focals_raw[i])[0]))

        return depth_maps, focal_lengths


class CuboidPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sam_backend: SamBackend | None = None
        self.depth_backend: DepthBackend | None = None

    def _torch_dtype(self) -> "torch.dtype":
        import torch

        dtype_str = self.settings.model_dtype.lower()
        if dtype_str == "float16":
            return torch.float16
        if dtype_str == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def load_models(self) -> None:
        torch_dtype = self._torch_dtype()
        logger.info("loading_models", extra={"device": self.settings.device})
        started = time.perf_counter()

        # Prefer the Transformers backend for batched SAM3. If unavailable, fallback.
        try:
            self.sam_backend = Sam3TransformersBackend(
                model_id=self.settings.sam3_model_id,
                device=self.settings.device,
                torch_dtype=torch_dtype,
            )
            logger.info("sam3_backend_initialized", extra={"backend": "transformers"})
        except Exception as exc:
            logger.warning(
                "sam3_transformers_unavailable_fallback_to_official",
                extra={"error": str(exc)},
            )
            self.sam_backend = Sam3OfficialBackend()
            logger.info("sam3_backend_initialized", extra={"backend": "official"})

        self.depth_backend = DepthBackend(
            model_id=self.settings.depth_model_id,
            device=self.settings.device,
            torch_dtype=torch_dtype,
        )
        elapsed = time.perf_counter() - started
        logger.info("models_loaded", extra={"seconds": round(elapsed, 3)})

    def unload_models(self) -> None:
        self.sam_backend = None
        self.depth_backend = None

    @property
    def is_ready(self) -> bool:
        return self.sam_backend is not None and self.depth_backend is not None

    def run_batch_inference(
        self, images: list[InferenceImage], prompt: str, threshold: float | None = None
    ) -> list[dict[str, object]]:
        if not self.is_ready:
            raise RuntimeError("Pipeline models are not loaded")

        confidence_threshold = (
            threshold
            if threshold is not None
            else float(self.settings.confidence_threshold)
        )
        batch_size = int(self.settings.gpu_batch_size)
        logger.info(
            "inference_started",
            extra={
                "images_count": len(images),
                "batch_size": batch_size,
                "prompt": prompt,
                "threshold": confidence_threshold,
            },
        )

        all_results: list[dict[str, object]] = []
        image_offset = 0
        for chunk in _chunked(images, batch_size):
            chunk_images = [entry.image.convert("RGB") for entry in chunk]
            stage_start = time.perf_counter()

            assert self.sam_backend is not None
            assert self.depth_backend is not None
            detections_by_image = self.sam_backend.segment(
                chunk_images, prompt, confidence_threshold
            )
            depth_maps, focal_lengths = self.depth_backend.estimate(chunk_images)
            stage_elapsed = time.perf_counter() - stage_start
            logger.info(
                "chunk_inference_complete",
                extra={
                    "chunk_size": len(chunk),
                    "seconds": round(stage_elapsed, 3),
                },
            )

            for local_idx, (entry, detections) in enumerate(
                zip(chunk, detections_by_image, strict=True)
            ):
                try:
                    cuboids = self._detections_to_cuboids(
                        detections=detections,
                        depth_map=depth_maps[local_idx],
                        focal_length_px=focal_lengths[local_idx],
                        image_size=entry.image.size,
                        prompt=prompt,
                    )
                    all_results.append(
                        {
                            "image_index": image_offset + local_idx,
                            "filename": entry.filename,
                            "cuboids": cuboids,
                            "count": len(cuboids),
                            "error": None,
                        }
                    )
                except Exception as exc:
                    logger.exception(
                        "image_inference_failed",
                        extra={
                            "image_index": image_offset + local_idx,
                            "filename": entry.filename,
                        },
                    )
                    all_results.append(
                        {
                            "image_index": image_offset + local_idx,
                            "filename": entry.filename,
                            "cuboids": [],
                            "count": 0,
                            "error": str(exc),
                        }
                    )

            image_offset += len(chunk)

        return all_results

    def _detections_to_cuboids(
        self,
        detections: list[dict[str, np.ndarray | float]],
        depth_map: np.ndarray,
        focal_length_px: float,
        image_size: tuple[int, int],
        prompt: str,
    ) -> list[dict[str, object]]:
        width, height = image_size
        intrinsics = build_intrinsics(focal_length_px, width, height)
        cuboids: list[dict[str, object]] = []

        for object_index, det in enumerate(detections):
            mask = np.asarray(det["mask"], dtype=bool)
            if mask.shape != depth_map.shape:
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (depth_map.shape[1], depth_map.shape[0]), Image.NEAREST
                    ),
                    dtype=bool,
                )

            points_3d = backproject_depth_to_3d(depth_map, mask, intrinsics)
            if len(points_3d) < 10:
                continue
            points_3d = filter_outliers(points_3d, std_multiplier=2.0)
            if len(points_3d) < 10:
                continue

            obb = fit_oriented_bounding_box(points_3d)
            cuboids.append(
                {
                    "object_index": object_index,
                    "label": prompt,
                    "confidence": float(det["score"]),
                    "center": obb["center"].tolist(),
                    "dimensions": obb["dimensions"].tolist(),
                    "rotation_matrix": obb["rotation_matrix"].tolist(),
                    "rotation_quaternion": rotation_matrix_to_quaternion(
                        obb["rotation_matrix"]
                    ).tolist(),
                    "corners_3d": obb["corners_3d"].tolist(),
                }
            )
        return cuboids


def decode_uploads(raw_uploads: list[tuple[str, bytes]]) -> list[InferenceImage]:
    images: list[InferenceImage] = []
    for filename, payload in raw_uploads:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        images.append(InferenceImage(filename=filename, image=image))
    return images

