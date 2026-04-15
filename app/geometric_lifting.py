from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def build_intrinsics(
    focal_length_px: float, image_width: int, image_height: int
) -> np.ndarray:
    """Construct a 3x3 camera intrinsic matrix."""
    cx = image_width / 2.0
    cy = image_height / 2.0
    return np.array(
        [
            [focal_length_px, 0.0, cx],
            [0.0, focal_length_px, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def backproject_depth_to_3d(
    depth_map: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Convert masked depth pixels into a 3D point cloud in camera coordinates."""
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim > 2:
        mask = mask[0]
    v, u = np.where(mask)
    if len(v) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = depth_map[v, u]
    valid = np.isfinite(z) & (z > 0.0) & (z < 100.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    u, v, z = u[valid], v[valid], z[valid]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1).astype(np.float64)


def filter_outliers(points_3d: np.ndarray, std_multiplier: float = 2.0) -> np.ndarray:
    """Remove point outliers based on per-axis median and standard deviation."""
    if len(points_3d) == 0:
        return points_3d

    median = np.median(points_3d, axis=0)
    std = np.std(points_3d, axis=0)
    std = np.where(std == 0.0, 1e-6, std)

    keep = np.all(np.abs(points_3d - median) < (std_multiplier * std), axis=1)
    return points_3d[keep]


def fit_oriented_bounding_box(
    points_3d: np.ndarray, min_dim_ratio: float = 0.3
) -> dict[str, np.ndarray]:
    """Fit an oriented bounding box to a 3D point cloud using PCA.

    Because a single-view backprojection yields a surface (not a volume),
    the smallest PCA dimension is often near-zero.  ``min_dim_ratio``
    inflates any collapsed axis to at least that fraction of the largest
    axis so the cuboid has realistic depth.
    """
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]

    if np.linalg.det(eigenvectors) < 0.0:
        eigenvectors[:, -1] *= -1.0

    projected = centered @ eigenvectors
    min_vals = projected.min(axis=0)
    max_vals = projected.max(axis=0)
    dimensions = max_vals - min_vals

    max_dim = dimensions.max()
    if max_dim > 0:
        min_allowed = max_dim * min_dim_ratio
        dimensions = np.maximum(dimensions, min_allowed)

    local_center = (min_vals + max_vals) / 2.0
    center = centroid + eigenvectors @ local_center

    half = dimensions / 2.0
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local_corners = signs * half
    corners = (eigenvectors @ local_corners.T).T + center

    return {
        "center": center,
        "dimensions": dimensions,
        "rotation_matrix": eigenvectors,
        "corners_3d": corners,
    }


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
    rotation = Rotation.from_matrix(rotation_matrix)
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

