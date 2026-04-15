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
    points_3d: np.ndarray,
    min_dim_ratio: float = 0.3,
    gravity_aligned: bool = True,
) -> dict[str, np.ndarray]:
    """Fit an oriented bounding box to a 3D point cloud.

    When ``gravity_aligned`` is True the cuboid's Y axis is locked to the
    camera-Y direction (vertical in the image) so that top/bottom edges
    stay horizontal.  The heading (yaw) in the XZ ground plane is still
    estimated from the point distribution via PCA.

    ``min_dim_ratio`` inflates any collapsed axis to at least that fraction
    of the largest axis so the cuboid has realistic depth.
    """
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid

    if gravity_aligned:
        axes = _gravity_aligned_axes(centered)
    else:
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        axes = eigenvectors[:, order]

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


def _gravity_aligned_axes(centered: np.ndarray) -> np.ndarray:
    """Build a rotation matrix with Y locked to camera-vertical.

    Camera convention: X = right, Y = down, Z = forward.
    The cuboid's local axes are:
      col 0 – "width"  : heading direction in the XZ ground plane (PCA)
      col 1 – "height" : camera Y  (gravity / vertical)
      col 2 – "depth"  : orthogonal, completes the right-hand frame
    """
    y_axis = np.array([0.0, 1.0, 0.0])

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

