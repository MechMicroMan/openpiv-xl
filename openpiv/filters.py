"""Image inpainting and filtering utilities for OpenPIV.

This module provides functions for replacing NaN values and outliers
in velocity field arrays using iterative image inpainting algorithms.
"""

from typing import Literal, Tuple

import numpy as np

__licence__ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


def get_dist(kernel: np.ndarray, kernel_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate distance maps from kernel center for weighting schemes.

    Parameters
    ----------
    kernel : np.ndarray
        Kernel array (2D or 3D) to generate distance maps for.
    kernel_size : int
        Kernel radius (distance from center).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dist, dist_inv): Euclidean distances from center and inverse distances.

    Raises
    ------
    ValueError
        If kernel is not 2D or 3D.
    """
    n_dim = len(kernel.shape)
    
    if n_dim == 2:
        xs, ys = np.indices(kernel.shape)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2)
        dist_inv = np.sqrt(2) * kernel_size - dist
    elif n_dim == 3:
        xs, ys, zs = np.indices(kernel.shape)
        dist = np.sqrt(
            (ys - kernel_size) ** 2
            + (xs - kernel_size) ** 2
            + (zs - kernel_size) ** 2
        )
        dist_inv = np.sqrt(3) * kernel_size - dist
    else:
        raise ValueError(f"Kernel must be 2D or 3D, got {n_dim}D")

    return dist, dist_inv

def replace_nans(
    array: np.ndarray,
    max_iter: int,
    tol: float,
    kernel_size: int = 2,
    method: Literal["localmean", "disk", "distance"] = "disk",
) -> np.ndarray:

    """Replace NaN elements using an iterative image inpainting algorithm.

    The algorithm replaces each NaN by a weighted average of non-NaN neighbors.
    Multiple iterations spread information from edges of missing regions until
    the variation falls below the specified tolerance.

    Parameters
    ----------
    array : np.ndarray (2D or 3D)
        Array containing NaN elements to be replaced. If a masked array
        (numpy.ma.MaskedArray), the mask is reapplied after replacement.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance threshold for convergence (MSE of replaced elements).
    kernel_size : int, optional
        Kernel radius, by default 2. The kernel size becomes (2*kernel_size+1).
    method : {"localmean", "disk", "distance"}, optional
        Weighting method for neighboring elements:
        - "localmean": Square kernel with uniform weights = n/((2*k+1)²-1)
        - "disk": Circular kernel with uniform weights inside radius
        - "distance": Circular inverse-distance weighting (closer = higher weight)
        Default is "disk".

    Returns
    -------
    np.ndarray
        Copy of input array with NaN elements replaced.

    Raises
    ------
    ValueError
        If method is not one of {"localmean", "disk", "distance"}.

    """

    kernel_size = int(kernel_size)
    filled = array.copy()
    
    # Find indices where array is NaN
    nan_indices = np.array(np.nonzero(np.isnan(array))).T.astype(int)
    n_nans = len(nan_indices)
    
    # Early return if no NaN values present
    if n_nans == 0:
        return filled
    
    n_dim = len(array.shape)

    # Generate the kernel based on method
    kernel_shape = [2 * kernel_size + 1] * n_dim
    kernel = np.zeros(kernel_shape, dtype=float)
    
    if method == "localmean":
        kernel += 1.0
    elif method == "disk":
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = 1.0
    elif method == "distance":
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = dist_inv[dist <= kernel_size]
    else:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: 'localmean', 'disk', 'distance'."
        )

    # Arrays to track replaced values and check convergence
    replaced_new = np.zeros(n_nans)
    replaced_old = np.zeros(n_nans)

    # Iteratively fill NaN values until convergence
    for iteration in range(max_iter):
        # For each NaN element, compute weighted average of neighbors
        for k in range(n_nans):
            # Position of the current NaN element
            nan_pos = nan_indices[k]
            replaced_new[k] = 0.0

            # Generate indices for the convolution window
            window_ranges = [range(pos - kernel_size, pos + kernel_size + 1) 
                           for pos in nan_pos]
            slice_indices = np.array(np.meshgrid(*window_ranges, indexing='ij'))

            # Create mask for indices within array bounds
            boundary_mask = np.ones(slice_indices.shape[1:], dtype=bool)
            for dim in range(n_dim):
                boundary_mask &= (
                    (slice_indices[dim] >= 0) & 
                    (slice_indices[dim] < array.shape[dim])
                )

            # Extract window values and corresponding kernel weights
            window_values = filled[tuple(slice_indices[:, boundary_mask])]
            kernel_weights = kernel[boundary_mask]

            # Compute weighted average (ignoring NaN values)
            valid_mask = ~np.isnan(window_values)
            valid_weight_sum = np.sum(kernel_weights[valid_mask])

            if valid_weight_sum > 0:
                replaced_new[k] = np.sum(window_values[valid_mask] * 
                                        kernel_weights[valid_mask]) / valid_weight_sum
            else:
                replaced_new[k] = np.nan

        # Update all NaN values with new estimates
        filled[tuple(nan_indices.T)] = replaced_new

        # Check convergence: if MSE is below tolerance, stop
        mse = np.mean((replaced_new - replaced_old) ** 2)
        if mse < tol:
            break
        
        replaced_old = replaced_new.copy()

    return filled

def replace_outliers(
    u: np.ndarray,
    v: np.ndarray,
    flags: np.ndarray,
    method: Literal["localmean", "disk", "distance"] = "localmean",
    max_iter: int = 5,
    tol: float = 1e-3,
    kernel_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace invalid velocity vectors using iterative image inpainting.

    Replaces flagged vectors in u and v velocity fields by weighted averages
    of valid neighboring vectors. The process iterates until the variation
    of replaced values falls below the tolerance threshold.

    Parameters
    ----------
    u : np.ndarray (2D or 3D)
        u velocity component field.
    v : np.ndarray (2D or 3D)
        v velocity component field.
    flags : np.ndarray (bool)
        Boolean array marking positions with invalid vectors.
    method : {"localmean", "disk", "distance"}, optional
        Kernel type for inpainting, by default "localmean".
    max_iter : int, optional
        Maximum number of iterations, by default 5.
    tol : float, optional
        Convergence tolerance (MSE), by default 1e-3.
    kernel_size : int, optional
        Kernel radius, by default 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (uf, vf): Repaired u and v velocity component fields.

    Notes
    -----
    This function modifies the input arrays by setting flagged positions to NaN
    before inpainting. Consider passing copies if you need to preserve originals.

    """

    # Mark invalid positions as NaN
    u[flags] = np.nan
    v[flags] = np.nan

    # Repair both velocity components independently
    uf = replace_nans(
        u, max_iter=max_iter, tol=tol, kernel_size=kernel_size, method=method
    )
    vf = replace_nans(
        v, max_iter=max_iter, tol=tol, kernel_size=kernel_size, method=method
    )

    return uf, vf
