# File:       PixelGrid.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-03
import numpy as np

def make_pixel_grid(xlims, zlims, nx, nz):
    """
    Generate a Cartesian pixel grid based on input parameters.
    The output has shape (nx, nz, 3).
    
    INPUTS
    xlims   Azimuthal limits of pixel grid ([xmin, xmax])
    zlims   Depth limits of pixel grid ([zmin, zmax])
    nx      Pixel spacing in azimuth
    nz      Pixel spacing in depth

    OUTPUTS
    grid    Pixel grid of size (nx, nz, 3)
    """
    dx = (xlims[1] - xlims[0]) / (nx - 1) if nx > 1 else (xlims[1] - xlims[0])
    dz = (zlims[1] - zlims[0]) / (nz - 1) if nz > 1 else (zlims[1] - zlims[0])
    x = np.linspace(xlims[0], xlims[1], nx)
    z = np.linspace(zlims[0], zlims[1], nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    yy = np.zeros_like(xx)
    iq_dimension = np.zeros_like(xx)
    grid = np.stack((xx, yy, zz, iq_dimension), axis=-1)
    return grid, dx, dz