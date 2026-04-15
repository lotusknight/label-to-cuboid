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


def estimate_heading_from_mask(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray | None:
    """Estimate a vehicle's 3D heading from its 2D mask shape.

    Runs 2D PCA on the mask pixels to find the major axis, picks two
    representative points along that axis, back-projects them into 3D
    using their actual depth values, and returns the 3D direction
    projected onto the XZ ground plane.  Much more stable than running
    PCA on the noisy 3D point cloud.
    """
    vs, us = np.where(mask)
    if len(vs) < 5:
        return None

    uv = np.column_stack([us.astype(np.float64), vs.astype(np.float64)])
    center_2d = uv.mean(axis=0)
    cov2d = np.cov(uv, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    major = eigvecs[:, eigvals.argsort()[-1]]

    proj = (uv - center_2d) @ major
    p10 = int(np.percentile(np.arange(len(proj)), 10, method="nearest"))
    p90 = int(np.percentile(np.arange(len(proj)), 90, method="nearest"))
    order = proj.argsort()
    idx_lo, idx_hi = order[max(p10, 0)], order[min(p90, len(order) - 1)]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    pts_3d = []
    for idx in (idx_lo, idx_hi):
        u_px, v_px = float(us[idx]), float(vs[idx])
        z = float(depth_map[int(v_px), int(u_px)])
        if not (np.isfinite(z) and z > 0):
            return None
        x3 = (u_px - cx) * z / fx
        z3 = z
        pts_3d.append(np.array([x3, 0.0, z3]))

    direction = pts_3d[1] - pts_3d[0]
    direction[1] = 0.0
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    return direction / norm


def fit_oriented_bounding_box(
    points_3d: np.ndarray,
    min_dim_ratio: float = 0.3,
    heading_hint: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fit a gravity-aligned oriented bounding box to a 3D point cloud.

    Y axis is locked to camera-vertical so top/bottom edges stay
    horizontal.  If ``heading_hint`` (a unit vector in the XZ plane)
    is provided it is used for the forward direction; otherwise the
    heading is estimated from 3D PCA on the XZ projection (noisier).

    ``min_dim_ratio`` inflates any collapsed axis to at least that
    fraction of the largest axis so the cuboid has realistic depth.
    """
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid
    axes = _gravity_aligned_axes(centered, heading_hint=heading_hint)

    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0

    projected = centered @ axes
    min_vals = projected.min(axis=0)
    max_vals = projected.max(axis=0)
    dimensions = max_vals - min_vals

    max_dim = dimensions.max()
    if max_dim > 0:
        min_allowed = max_dim * min_dim_ratio
        dimensions = np.maximum(dimensions, min_allowed)

    local_center = (min_vals + max_vals) / 2.0
    center = centroid + axes @ local_center

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
    corners = (axes @ local_corners.T).T + center

    return {
        "center": center,
        "dimensions": dimensions,
        "rotation_matrix": axes,
        "corners_3d": corners,
    }


def _gravity_aligned_axes(
    centered: np.ndarray,
    heading_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Build a rotation matrix with Y locked to camera-vertical.

    If ``heading_hint`` is given (a unit vector in the XZ plane) it is
    used directly as the forward direction.  Otherwise falls back to
    PCA on the XZ projection of the 3D points.
    """
    y_axis = np.array([0.0, 1.0, 0.0])

    if heading_hint is not None:
        forward = heading_hint.copy()
        forward[1] = 0.0
        norm = np.linalg.norm(forward)
        if norm > 1e-8:
            forward /= norm
        else:
            forward = np.array([0.0, 0.0, 1.0])
    else:
        xz = centered[:, [0, 2]]
        if xz.shape[0] < 2:
            forward = np.array([0.0, 0.0, 1.0])
        else:
            cov2d = np.cov(xz, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov2d)
            principal = eigvecs[:, eigvals.argsort()[-1]]
            forward = np.array([principal[0], 0.0, principal[1]])
            norm = np.linalg.norm(forward)
            if norm < 1e-8:
                forward = np.array([0.0, 0.0, 1.0])
            else:
                forward /= norm

    side = np.cross(y_axis, forward)
    norm = np.linalg.norm(side)
    if norm < 1e-8:
        side = np.array([1.0, 0.0, 0.0])
    else:
        side /= norm
    forward = np.cross(side, y_axis)

    axes = np.column_stack([side, y_axis, forward])
    return axes


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
    rotation = Rotation.from_matrix(rotation_matrix)
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

